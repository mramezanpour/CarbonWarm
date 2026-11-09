"""Residency, placement and deferral policies.

Baselines
  Fixed-KA             10-min keep-alive, then SSD; nearest site
  Histogram            hybrid-histogram keep-alive and pre-warming (Shahrad et al., ATC'20); nearest site
  ServerlessLLM        10-min keep-alive, then LRU DRAM cache; lowest estimated TTFT (Fu et al., OSDI'24)
  Breakeven-E          energy breakeven keep-alive T* = E_reload / P_ctx; nearest site
  MultiState-PD        multi-state power-down, lower envelope over GPU/DRAM/SSD/remote (Irani et al.)
  Carbon-Routing       lowest serving carbon within the SLO, assumes warm models; 10-min keep-alive
  Carbon-Routing+CS    as Carbon-Routing, with load time and load carbon of cold starts
  Carbon-Routing+DRAM  as Carbon-Routing+CS, with a ServerlessLLM-style DRAM cache
Proposed
  CarbonWarm           carbon-weighted multi-tier ladder, warmth-aware placement with an adaptive SLO price,
                       green pre-loading and carbon-aware deferral
Ablations              CarbonWarm without one component (ABLATIONS)
Clairvoyant            per-gap greedy residency with knowledge of all future dispatches (reference, not a bound)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .carbon import KWH_J
from .sim import TIER_DRAM, TIER_GPU, TIER_REMOTE, TIER_SSD, ModelSite, Replica, Simulator


@dataclass
class _Step:
    kind: str          # 'gpu' | 'dram->ssd' | 'ssd->remote'
    ref: object        # replica id or ModelSite
    ver: int
    p_hi: float
    p_lo: float
    thr: float
    acc: float
    last: float
    to: str = "ssd"


class Policy:
    def __init__(self, name, placement="nearest", residency="fixed", gpu_demote_to="ssd",
                 keepalive_s=600.0, defers=False, use_dram=True, use_pred=True, use_forecast=True,
                 energy_only=False, adaptive_mu=False, replay=None, use_forecast_res=None,
                 use_forecast_defer=None):
        self.name = name
        self.adaptive_mu = adaptive_mu
        self.mu = None
        self.win = deque(maxlen=400)  # recent interactive SLO outcomes
        self.n_obs = 0
        self.tin_hist = deque(maxlen=512)   # recent input lengths (online, no look-ahead)
        self._vcache = {}
        self.placement = placement
        self.residency = residency
        self.gpu_demote_to = gpu_demote_to
        self.keepalive_s = keepalive_s
        self.defers = defers
        self.use_dram = use_dram
        self.use_pred = use_pred
        self.use_forecast = use_forecast
        # the forecast is used in two places that can be ablated separately
        self.use_forecast_res = use_forecast if use_forecast_res is None else use_forecast_res
        self.use_forecast_defer = use_forecast if use_forecast_defer is None else use_forecast_defer
        self.mu_trace = []   # (time, mu) after every update
        self.energy_only = energy_only
        self.replay = replay
        self.buffer: Dict[int, List[list]] = {}

    # ------------------------------------------------------------------ init
    def init(self, sim: Simulator):
        self.I_const = float(np.mean([s.carbon.mean for s in sim.sites]))
        self.mu = sim.sp.mu_g_per_violation
        self.typ_tin = 512  # prior until enough requests have been seen
        if self.residency == "oracle":
            site, disp = self.replay["site"], self.replay["disp"]
            for m in range(len(sim.models)):
                for s in range(len(sim.sites)):
                    mask = (sim.wl.model == m) & (site == s)
                    sim.ms[m][s].future = np.sort(disp[mask])

    # ---------------------------------------------------- SLO dual variable (mu)
    def observe(self, sim, ok: bool):
        """Online (multiplicative) dual ascent for the chance constraint
        P(TTFT <= SLO) >= target: mu = price in grams of CO2 per SLO violation."""
        if not self.adaptive_mu:
            return
        self.win.append(1.0 if ok else 0.0)
        self.n_obs += 1
        if self.n_obs % 25 == 0 and len(self.win) >= 50:
            # internal target = required target + safety margin (the time-average of a bursty
            # process controlled online otherwise settles slightly below the target)
            tgt = min(sim.sp.slo_target_pct + sim.sp.slo_margin_pct, 99.9) / 100.0
            err = tgt - float(np.mean(self.win))  # > 0: too many violations -> raise mu
            self.mu = float(np.clip(self.mu * np.exp(sim.sp.mu_gain * err / max(1 - tgt, 1e-3)), 1e-3,
                                    sim.sp.mu_max))
            self.mu_trace.append((sim.t, self.mu, float(np.mean(self.win))))

    def mu_val(self, sim):
        return self.mu if self.mu is not None else sim.sp.mu_g_per_violation

    def note_request(self, sim, i):
        """Record the input length of an arriving request (used to estimate violation risk)."""
        self.tin_hist.append(int(sim.wl.tin[i]))
        if len(self.tin_hist) % 128 == 0:
            self._vcache.clear()

    def violates(self, sim, m, s, tier) -> float:
        """Probability that the next request misses the TTFT SLO if model m has to be loaded from
        `tier` at site s: share of recent input lengths for which load + prefill + RTT > SLO.
        (A fractional estimate lets mu act continuously; a yes/no estimate based on a typical
        prompt ignores the long-prompt tail and makes the SLO price ineffective.)"""
        if tier == TIER_GPU:
            return 0.0
        key = (m, s, tier)
        v = self._vcache.get(key)
        if v is None:
            tins = np.asarray(self.tin_hist, float) if len(self.tin_hist) >= 32 else np.array([self.typ_tin], float)
            prefill = tins / (sim.sites[s].gt.prefill_tps_7b / sim.speed(m))
            typ_rtt = float(np.min(sim.rtt[:, s]))
            v = float(np.mean(sim.load_time(m, s, tier) + prefill + typ_rtt > sim.sp.slo_ttft_s))
            self._vcache[key] = v
        return v

    def intensity(self, sim, s, t):
        return self.I_const if self.energy_only else sim.sites[s].carbon.I(t)

    def intensity_hat(self, sim, s, t):
        if self.energy_only:
            return self.I_const
        c = sim.sites[s].carbon
        return c.forecast(t, sim.sp.forecast_horizon_h) if self.use_forecast_res else c.I(t)

    # ------------------------------------------------------------- placement
    def candidates(self, sim, m):
        S = len(sim.sites)
        c = [s for s in range(S) if sim.fits[m][s] and sim.can_host(m, s)]
        if not c:
            c = [s for s in range(S) if sim.fits[m][s]]
        return c

    def place(self, sim: Simulator, i, t):
        m = sim.wl.model[i]
        reg = sim.wl.region[i]
        self.note_request(sim, i)
        cands = self.candidates(sim, m)
        mode = self.placement
        if mode == "replay":
            s = int(self.replay["site"][i])
            if s >= 0:
                return s
            mode = "nearest"
        if mode == "nearest":
            h = sim.home[reg]
            if h in cands:
                return h
            return min(cands, key=lambda s: sim.rtt[reg][s])
        if mode == "locality":
            return min(cands, key=lambda s: (sim.est_ttft(i, s), sim.rtt[reg][s]))
        slo = sim.sp.slo_ttft_s
        if mode == "carbon_greedy":
            ok = [s for s in cands if sim.rtt[reg][s] + sim.prefill_time(i, s) <= slo]
            if not ok:
                return min(cands, key=lambda s: sim.est_ttft(i, s))
            return min(ok, key=lambda s: sim.marginal_carbon(i, s, include_load=False))
        if mode == "carbon_greedy_cold":
            # baseline: carbon-aware routing that sees the cold start (load time in the SLO check,
            # load carbon in the cost)
            ok = [s for s in cands if sim.est_ttft(i, s) <= slo]
            if not ok:
                return min(cands, key=lambda s: sim.est_ttft(i, s))
            return min(ok, key=lambda s: (sim.marginal_carbon(i, s, include_load=True), sim.rtt[reg][s]))
        if mode == "carbonwarm":
            return self._place_carbonwarm(sim, i, t, cands, reg, slo)
        raise ValueError(mode)

    def _place_carbonwarm(self, sim, i, t, cands, reg, slo):
        """Warmth-aware Lagrangian placement.

        cost(s) = serving carbon + [cold start: load carbon + expected idle holding]
                  + mu * 1[estimated TTFT at s > SLO]
        mu is the SLO dual variable, so slack in the SLO is automatically spent on greener
        sites and tightened when attainment drops below the target.
        Green pre-warm: if a clearly greener site is only cold, start loading it in the
        background when the model is busy enough to amortise the load.
        """
        m = sim.wl.model[i]
        I = {s: self.intensity(sim, s, t) for s in cands}
        mu = self.mu_val(sim)
        cost = {s: self._cw_cost(sim, i, s, I[s]) + mu * (sim.est_ttft(i, s) > slo) for s in cands}
        choice = min(cands, key=lambda s: (cost[s], sim.rtt[reg][s]))
        if sim.sp.green_prewarm:
            self._green_prewarm(sim, i, t, cands, choice, reg, slo, I)
        hist = self.__dict__.setdefault("arr_hist", {}).setdefault(m, [])
        hist.append(t)
        if len(hist) > 64:
            del hist[:32]
        return choice

    def _cw_cost(self, sim, i, s, I_s):
        """Marginal carbon of serving i at s, incl. the ski-rental cost of a cold start.

        A cold start costs its load carbon now and -- by the ski-rental argument -- about one
        more threshold's worth of idle holding afterwards (load carbon + mu * P(violation)).
        """
        m = sim.wl.model[i]
        ms = sim.ms[m][s]
        g = sim.marginal_carbon(i, s, I_s)
        if sim.ready_spare(ms) is None and not any(r.state == "loading" for r in ms.replicas):
            tier = sim.best_tier(ms)
            mu = self.mu_val(sim)
            g += 2.0 * sim.load_energy(m, s, tier) * I_s / KWH_J + mu * self.violates(sim, m, s, tier)
            if sim.sp.capacity_aware:
                g += self._displacement(sim, m, s, I_s, mu)
        return g

    def _displacement(self, sim, m, s, I_s, mu):
        """a cold placement at a site whose memory is full displaces another model, which must later be
        reloaded from a colder tier. The displaced model is not known in advance; its cost is approximated
        with the requesting model's own reload costs (same order of size)."""
        site = sim.sites[s]
        ms = sim.ms[m][s]
        size = sim.models[m].size_GB
        extra = 0.0
        need = sim.need_GB(m)
        if not any(gp.mem_free >= need for gp in site.gpus):          # evicts an idle GPU replica
            extra += sim.load_energy(m, s, TIER_DRAM) * I_s / KWH_J + mu * self.violates(sim, m, s, TIER_DRAM)
        if not ms.has_dram and size <= site.spec.dram_GB:
            # a new warm copy at this site will later occupy DRAM until the ladder's DRAM->SSD step fires;
            # by the ski-rental argument that holding costs at most one buy cost of that step
            extra += (sim.load_energy(m, s, TIER_SSD) - sim.load_energy(m, s, TIER_DRAM)) * I_s / KWH_J + \
                mu * max(self.violates(sim, m, s, TIER_SSD) - self.violates(sim, m, s, TIER_DRAM), 0.0)
        if not ms.has_dram and site.dram_used + size > site.spec.dram_GB:  # will push a model out of DRAM
            extra += (sim.load_energy(m, s, TIER_SSD) - sim.load_energy(m, s, TIER_DRAM)) * I_s / KWH_J + \
                mu * max(self.violates(sim, m, s, TIER_SSD) - self.violates(sim, m, s, TIER_DRAM), 0.0)
        return extra

    def _green_prewarm(self, sim, i, t, cands, choice, reg, slo, I):
        m = sim.wl.model[i]
        hist = self.__dict__.get("arr_hist", {}).get(m, [])
        if len(hist) < 8 or t - hist[0] <= 0:
            return
        rate = len(hist) / (t - hist[0])
        steady = {s: sim.marginal_carbon(i, s, I[s], include_load=False) for s in cands}
        warm_or_loading = lambda s: bool(sim.ms[m][s].replicas)
        green = [s for s in cands if not warm_or_loading(s) and
                 sim.rtt[reg][s] + sim.prefill_time(i, s) <= slo and steady[s] < 0.9 * steady[choice]]
        if not green:
            return
        s = min(green, key=lambda x: steady[x])
        if sim.sp.prewarm_no_evict:  # a speculative load must not evict other models' replicas
            need = sim.need_GB(m)
            if not any(g.mem_free >= need for g in sim.sites[s].gpus):
                return
            st = sim.sites[s]   # ... nor push another model out of the site's DRAM cache later
            if not sim.ms[m][s].has_dram and st.dram_used + sim.models[m].size_GB > st.spec.dram_GB:
                return
        save = (steady[choice] - steady[s]) * rate * 1800.0  # expected saving over ~30 min
        load_g = sim.load_energy(m, s, sim.best_tier(sim.ms[m][s])) * I[s] / KWH_J
        if save > 2.0 * load_g:
            sim.prewarm(m, s, t)

    # -------------------------------------------------------------- deferral
    def on_deferrable(self, sim: Simulator, i, t):
        if self.placement == "replay":  # oracle: reproduce the recorded dispatch time and site
            td, s = self.replay["disp"][i], int(self.replay["site"][i])
            if not np.isfinite(td) or s < 0:
                sim.dispatch(i, self.place(sim, i, t), t)
            else:
                sim.call_at(max(td, t), _replay_dispatch, i, s)
            return
        m = sim.wl.model[i]
        cands = [s for s in range(len(sim.sites)) if sim.fits[m][s]]
        D = sim.wl.deadline[i]
        worst = max(sim.prefill_time(i, s) + sim.wl.tout[i] * sim.step_time(m, s, 8) for s in cands)
        latest = min(D - 1800.0 - worst, sim.sp.duration_s - 60.0)
        if latest <= t:
            self._dispatch_deferred(sim, t, [i, t, None, None, False])
            return
        if not self.use_forecast_defer:
            # no forecast: reactive rule -- run when some site is in its greenest quartile of the
            # past 24 h (known history only), otherwise at the latest safe time
            thr = min(self._hist_q25(sim, s, t) for s in cands)
            item = [i, latest, None, thr, False, latest]
            self.buffer.setdefault(m, []).append(item)
            self._poll(sim, t, item)
            return
        best = None
        for s in cands:
            ts, Is = sim.sites[s].carbon.best_start(t, latest)
            _, _, e, base = sim.service(i, s, 8)
            score = Is * (e + base)
            if best is None or score < best[0]:
                best = (score, ts, s, Is)
        item = [i, best[1], best[2], best[3], False]
        self.buffer.setdefault(m, []).append(item)
        sim.call_at(best[1], self._fire, item)

    @staticmethod
    def _hist_q25(sim, s, t):
        c = sim.sites[s].carbon
        h = c._h(t)
        past = c.values[max(0, h - 24):h + 1]
        return float(np.percentile(past, 25))

    def _poll(self, sim, t, item):
        if item[4]:
            return
        m = sim.wl.model[item[0]]
        cands = [s for s in range(len(sim.sites)) if sim.fits[m][s]]
        i = item[0]
        best = min(cands, key=lambda s: self._cw_cost(sim, i, s, sim.sites[s].carbon.I(t)))
        if sim.sites[best].carbon.I(t) <= self._hist_q25(sim, best, t) or t >= item[5] - 1e-6:
            self._dispatch_deferred(sim, t, item, site=best)
            return
        nxt = min(sim.sites[0].carbon.next_hour_boundary(t), item[5])
        sim.call_at(nxt, _poll_cb, self, item)

    def _fire(self, sim, t, item):
        if not item[4]:
            self._dispatch_deferred(sim, t, item)

    def _dispatch_deferred(self, sim, t, item, site=None):
        i = item[0]
        item[4] = True
        m = sim.wl.model[i]
        if site is None:
            cands = self.candidates(sim, m)
            site = min(cands, key=lambda s: self._cw_cost(sim, i, s, self.intensity(sim, s, t)))
        sim.dispatch(i, site, t)
        lst = self.buffer.get(m)
        if lst is not None and item in lst:
            lst.remove(item)

    def piggyback(self, sim: Simulator, r: Replica, t) -> bool:
        lst = self.buffer.get(r.m)
        if not lst:
            return False
        I_now = sim.sites[r.s].carbon.I(t)
        B = sim.sites[r.s].gt.max_batch
        n = 0
        for item in list(lst):
            if n >= B:
                break
            if item[4] or not sim.fits[r.m][r.s]:
                continue
            if I_now <= item[3] * 1.15:
                self._dispatch_deferred(sim, t, item, site=r.s)
                n += 1
        return n > 0

    # ------------------------------------------------------------- residency
    def on_replica_idle(self, sim: Simulator, r: Replica, t):
        if self.defers and self.piggyback(sim, r, t):
            if r.idle_since is None:
                return
        ms = sim.ms[r.m][r.s]
        if r.prewarmed and r.keep_until is not None:
            sim.call_at(max(r.keep_until, t), _demote_if, r.id, r.ver, None)
            return
        res = self.residency
        if res in ("ladder", "oracle"):
            other_idle = [o for o in ms.replicas if o is not r and o.state == "ready" and o.n_active == 0]
            if other_idle:  # scale in: one idle replica is enough
                sim.demote_replica(r, t, to=self.gpu_demote_to)
                return
        if res == "fixed":
            sim.call_at(t + self.keepalive_s, _demote_if, r.id, r.ver, None)
        elif res == "breakeven":
            gt = sim.sites[r.s].gt
            p_park = gt.p_ctx_W / max(len(r.gpu.residents), 1)
            T = sim.load_energy(r.m, r.s, TIER_SSD) / max(p_park, 1e-9)
            sim.call_at(t + T, _demote_if, r.id, r.ver, None)
        elif res == "histogram":
            self._histogram(sim, r, ms, t)
        elif res == "multistate":
            self._multistate(sim, r, ms, t)
        elif res == "ladder":
            self._start_gpu_step(sim, r, ms, t)
        elif res == "oracle":
            self._oracle(sim, r, ms, t)
        else:
            raise ValueError(res)

    def on_offgpu(self, sim: Simulator, ms: ModelSite, t):
        if self.residency != "ladder":
            return
        if ms.has_dram:
            self._start_cache_step(sim, ms, "dram->ssd", t)
        elif ms.has_ssd:
            self._start_cache_step(sim, ms, "ssd->remote", t)

    # -- Shahrad et al. hybrid histogram ----------------------------------
    def _histogram(self, sim, r, ms, t):
        if len(ms.gaps) < 10:
            sim.call_at(t + self.keepalive_s, _demote_if, r.id, r.ver, None)
            return
        p5, p99 = np.percentile(np.asarray(ms.gaps), [5, 99])
        pre = 0.9 * p5
        ka = min(1.1 * p99, 4 * 3600.0)
        if pre >= 120.0:
            sim.demote_replica(r, t)
            sim.call_at(t + pre, _prewarm, r.m, r.s, t + max(ka, pre + 60.0))
        else:
            sim.call_at(t + max(ka, 60.0), _demote_if, r.id, r.ver, None)

    # -- Irani et al. multi-state power-down (lower-envelope algorithm) ------
    def _multistate(self, sim, r, ms, t):
        """Deterministic lower-envelope algorithm: state k has line c_k(x) = e_k + h_k x (reload energy
        plus holding power x idle time). The algorithm is always in the state of the lower envelope
        at the current idle time x; this is 2-competitive in energy for any number of states."""
        size = sim.models[r.m].size_GB
        en = sim.en
        gt = sim.sites[r.s].gt
        states = [("gpu", gt.p_ctx_W / max(len(r.gpu.residents), 1), 0.0)]
        if self.use_dram and size <= sim.sites[r.s].spec.dram_GB:
            states.append(("dram", en.p_dram_W_per_GB * size, sim.load_energy(r.m, r.s, TIER_DRAM)))
        states.append(("ssd", en.p_ssd_W_per_GB * size, sim.load_energy(r.m, r.s, TIER_SSD)))
        states.append(("remote", 0.0, sim.load_energy(r.m, r.s, TIER_REMOTE)))
        plan, k, x = [], 0, 0.0
        while k < len(states) - 1:
            best = None
            for j in range(k + 1, len(states)):
                dh = states[k][1] - states[j][1]
                if dh <= 0:
                    continue
                xj = (states[j][2] - states[k][2]) / dh
                if best is None or xj < best[0] - 1e-12 or (abs(xj - best[0]) <= 1e-12 and j > best[1]):
                    best = (max(xj, x), j)
            if best is None:
                break
            x, k = best
            plan.append((x, states[k][0]))
        if not plan:
            return
        x0, first = plan[0]
        sim.call_at(t + x0, _ms_first, r.id, r.ver, first, [(t + xx, nm) for xx, nm in plan[1:]])

    # -- CarbonWarm ladder --------------------------------------------------
    def _beta(self, sim, ms, thr1, rate_hat):
        if not self.use_pred:
            return 1.0
        if len(ms.gaps) < 5 or rate_hat <= 0:
            return 1.0
        t_be = thr1 / rate_hat
        gaps = np.asarray(ms.gaps)
        if sim.sp.prediction_noise > 0:
            gaps = gaps * sim.rng.lognormal(0.0, sim.sp.prediction_noise, len(gaps))
        p_long = float((gaps >= t_be).mean())  # predicted P(idle period outlasts the threshold)
        lam = sim.sp.trust_lambda
        if p_long >= 0.5:
            return lam          # long idle expected: demote early
        if p_long <= 0.1:
            return 1.0 / lam    # next request expected soon: stay longer
        return 1.0

    def _threshold(self, sim, ms, s, t, dE, dviol, p_hi, p_lo):
        """Ski-rental 'buy' cost of stepping down one tier: extra reload carbon (with the
        forecast intensity) + mu * extra SLO-violation probability of the next request."""
        I_hat = self.intensity_hat(sim, s, t)
        thr1 = dE * I_hat / KWH_J + self.mu_val(sim) * max(dviol, 0.0)
        rate_hat = (p_hi - p_lo) * I_hat / KWH_J
        return self._beta(sim, ms, thr1, rate_hat) * thr1

    def _start_gpu_step(self, sim, r, ms, t):
        size = sim.models[r.m].size_GB
        en = sim.en
        gt = sim.sites[r.s].gt
        to = "dram" if (self.use_dram and size <= sim.sites[r.s].spec.dram_GB) else "ssd"
        tier = TIER_DRAM if to == "dram" else TIER_SSD
        p_hi = gt.p_ctx_W / max(len(r.gpu.residents), 1)
        if to == "dram":
            p_lo = 0.0 if ms.has_dram else en.p_dram_W_per_GB * size
        else:
            p_lo = 0.0 if ms.has_ssd else en.p_ssd_W_per_GB * size
        dE = sim.load_energy(r.m, r.s, tier)
        dviol = self.violates(sim, r.m, r.s, tier)
        thr = self._threshold(sim, ms, r.s, t, dE, dviol, p_hi, p_lo)
        st = _Step("gpu", r.id, r.ver, p_hi, p_lo, thr, 0.0, t, to)
        self._check(sim, t, st, r.s)

    def _start_cache_step(self, sim, ms, kind, t):
        size = sim.models[ms.m].size_GB
        en = sim.en
        if kind == "dram->ssd":
            p_hi, p_lo = en.p_dram_W_per_GB * size, 0.0
            a, b = TIER_DRAM, TIER_SSD
        else:
            p_hi, p_lo = en.p_ssd_W_per_GB * size, 0.0
            a, b = TIER_SSD, TIER_REMOTE
        dE = sim.load_energy(ms.m, ms.s, b) - sim.load_energy(ms.m, ms.s, a)
        dviol = self.violates(sim, ms.m, ms.s, b) - self.violates(sim, ms.m, ms.s, a)
        thr = self._threshold(sim, ms, ms.s, t, dE, dviol, p_hi, p_lo)
        st = _Step(kind, ms, ms.ladder_ver, p_hi, p_lo, thr, 0.0, t)
        self._check(sim, t, st, ms.s)

    def _valid(self, sim, st):
        if st.kind == "gpu":
            r = sim.replicas.get(st.ref)
            return r is not None and r.ver == st.ver and r.idle_since is not None
        ms = st.ref
        if ms.ladder_ver != st.ver or ms.replicas:
            return False
        return ms.has_dram if st.kind == "dram->ssd" else (ms.has_ssd and not ms.has_dram)

    def _check(self, sim, t, st, s):
        if not self._valid(sim, st):
            return
        I_prev = self.intensity(sim, s, st.last)
        st.acc += (st.p_hi - st.p_lo) * I_prev * (t - st.last) / KWH_J
        st.last = t
        if st.acc >= st.thr - 1e-12:
            if st.kind == "gpu":
                sim.demote_replica(sim.replicas[st.ref], t, to=st.to)
            else:
                ms = st.ref
                sim.demote_cache(ms, t, st.kind)
                if st.kind == "dram->ssd" and ms.has_ssd:
                    self._start_cache_step(sim, ms, "ssd->remote", t)
            return
        rate = (st.p_hi - st.p_lo) * self.intensity(sim, s, t) / KWH_J
        if rate <= 0:
            return
        tn = t + (st.thr - st.acc) / rate
        if not self.energy_only:
            tn = min(tn, sim.sites[s].carbon.next_hour_boundary(t))
        sim.call_at(max(tn, t + 1e-6), _check_cb, self, st, s)

    # -- clairvoyant oracle -------------------------------------------------
    def _oracle(self, sim, r, ms, t):
        fut = ms.future
        k = int(np.searchsorted(fut, t, side="right")) if fut is not None else 0
        if fut is None or k >= len(fut):
            sim.demote_replica(r, t, to="remote")
            return
        a = float(fut[k])
        c = sim.sites[r.s].carbon
        size = sim.models[r.m].size_GB
        en = sim.en
        gt = sim.sites[r.s].gt
        p_attr = gt.p_ctx_W / max(len(r.gpu.residents), 1)
        best = ("stay", c.grams_power(p_attr, t, a), None)
        opts = []
        if size <= sim.sites[r.s].spec.dram_GB:
            opts.append(("dram", TIER_DRAM, en.p_dram_W_per_GB * size + en.p_ssd_W_per_GB * size))
        opts.append(("ssd", TIER_SSD, en.p_ssd_W_per_GB * size))
        opts.append(("remote", TIER_REMOTE, 0.0))
        for name, tier, p_hold in opts:
            lt = sim.load_time(r.m, r.s, tier)
            tp = max(t, a - lt - 1.0)
            cost = c.grams_power(p_hold, t, tp) + c.grams_energy(sim.load_energy(r.m, r.s, tier), tp, tp + lt)
            if cost < best[1]:
                best = (name, cost, tp)
        if best[0] == "stay":
            return
        name, _, tp = best
        sim.demote_replica(r, t, to=name)
        if name == "ssd":
            sim.set_dram(ms, False, t)
        sim.call_at(tp, _prewarm, r.m, r.s, None)


# --------------------------------------------------------------------- callbacks
def _demote_if(sim, t, rid, ver, to):
    r = sim.replicas.get(rid)
    if r is not None and r.ver == ver and r.idle_since is not None:
        sim.demote_replica(r, t, to=to)


def _ms_first(sim, t, rid, ver, to, rest):
    r = sim.replicas.get(rid)
    if r is None or r.ver != ver or r.idle_since is None:
        return
    ms = sim.ms[r.m][r.s]
    if not sim.demote_replica(r, t, to=to):
        return
    if ms.replicas:
        return
    lv = ms.ladder_ver
    for tt, nm in rest:
        sim.call_at(tt, _ms_next, ms, lv, nm)


def _ms_next(sim, t, ms, lv, to):
    if ms.replicas or ms.ladder_ver != lv:
        return
    if to == "ssd":
        sim.demote_cache(ms, t, "dram->ssd")
    elif to == "remote":
        if ms.has_dram:
            sim.demote_cache(ms, t, "dram->ssd")
        sim.demote_cache(ms, t, "ssd->remote")


def _poll_cb(sim, t, pol, item):
    pol._poll(sim, t, item)


def _replay_dispatch(sim, t, i, s):
    sim.dispatch(i, s, t)


def _prewarm(sim, t, m, s, keep_until):
    sim.prewarm(m, s, t, keep_until)


def _check_cb(sim, t, pol, st, s):
    pol._check(sim, t, st, s)


# --------------------------------------------------------------------- registry
def _cw(name, **kw):
    base = dict(placement="carbonwarm", residency="ladder", gpu_demote_to="dram", defers=True,
                adaptive_mu=True)
    base.update(kw)
    return lambda replay=None: Policy(name, **base, replay=replay)


POLICIES = {
    "Fixed-KA": lambda replay=None: Policy("Fixed-KA", "nearest", "fixed", "ssd"),
    "Histogram": lambda replay=None: Policy("Histogram", "nearest", "histogram", "ssd"),
    "ServerlessLLM": lambda replay=None: Policy("ServerlessLLM", "locality", "fixed", "dram"),
    "Breakeven-E": lambda replay=None: Policy("Breakeven-E", "nearest", "breakeven", "ssd"),
    "Carbon-Routing": lambda replay=None: Policy("Carbon-Routing", "carbon_greedy", "fixed", "ssd"),
    "Carbon-Routing+CS": lambda replay=None: Policy("Carbon-Routing+CS", "carbon_greedy_cold", "fixed", "ssd"),
    "Carbon-Routing+DRAM": lambda replay=None: Policy("Carbon-Routing+DRAM", "carbon_greedy_cold", "fixed", "dram"),
    "MultiState-PD": lambda replay=None: Policy("MultiState-PD", "nearest", "multistate", "ssd"),
    "CarbonWarm": _cw("CarbonWarm"),
}

ABLATIONS = {
    "CW w/o DRAM tier": _cw("CW w/o DRAM tier", use_dram=False, gpu_demote_to="ssd"),
    "CW w/o prediction": _cw("CW w/o prediction", use_pred=False),
    "CW w/o res. forecast": _cw("CW w/o res. forecast", use_forecast_res=False),
    "CW w/o defer. forecast": _cw("CW w/o defer. forecast", use_forecast_defer=False),
    "CW w/o coord. placement": _cw("CW w/o coord. placement", placement="nearest"),
    "CW w/o deferral": _cw("CW w/o deferral", defers=False),
    "CW w/o adaptive mu": _cw("CW w/o adaptive mu", adaptive_mu=False),
    "CW energy-only": _cw("CW energy-only", energy_only=True, defers=False),
}

ORACLE = "Clairvoyant"   # kept the variable name ORACLE for backward compatibility


def make_policy(name, replay=None) -> Policy:
    if name in POLICIES:
        return POLICIES[name](replay)
    if name in ABLATIONS:
        return ABLATIONS[name](replay)
    if name == ORACLE:
        return Policy(ORACLE, placement="replay", residency="oracle", gpu_demote_to="ssd",
                      defers=True, replay=replay)
    raise KeyError(name)
