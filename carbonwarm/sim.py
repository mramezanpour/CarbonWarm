"""Event-driven simulator of serverless LLM inference on cloud and edge sites.

Sites own GPUs of one type, a host-DRAM checkpoint cache and a local SSD cache; a remote store holds every
checkpoint. A replica is one copy of a model on one GPU and serves up to max_batch concurrent requests.
Carbon is power times the grid intensity of the site, integrated exactly over hourly intervals.
"""
from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from .carbon import KWH_J, CarbonSeries, build_carbon
from .config import (CARBON_SCENARIOS, GPU_TYPES, HOME_EDGE, USER_REGIONS, SimParams, SiteSpec,
                     BARE_IDLE_W, HOST_BASE_W,
                     default_rtt, default_sites)
from .workload import Workload

EV_ARR, EV_DONE, EV_LOADED, EV_CALL, EV_SAMPLE = 0, 1, 2, 3, 4
TIER_GPU, TIER_DRAM, TIER_SSD, TIER_REMOTE = 0, 1, 2, 3
CATS = ["idle", "load", "serve", "dram", "ssd"]


class GPU:
    __slots__ = ("id", "site", "gt", "mem_free", "residents", "n_loading", "n_active", "last")

    def __init__(self, gid, site, gt):
        self.id, self.site, self.gt = gid, site, gt
        self.mem_free = gt.mem_GB
        self.residents = set()
        self.n_loading = 0
        self.n_active = 0
        self.last = 0.0


class Replica:
    __slots__ = ("id", "m", "s", "gpu", "state", "n_active", "queue", "ready_t", "idle_since",
                 "ver", "last_used", "need", "prewarmed", "keep_until", "src")

    def __init__(self, rid, m, s, gpu, need):
        self.id, self.m, self.s, self.gpu, self.need = rid, m, s, gpu, need
        self.state = "loading"
        self.n_active = 0
        self.queue = deque()
        self.ready_t = math.inf
        self.idle_since = None
        self.ver = 0
        self.last_used = 0.0
        self.prewarmed = False
        self.keep_until = None
        self.src = TIER_REMOTE


class ModelSite:
    __slots__ = ("m", "s", "has_dram", "has_ssd", "t_dram", "t_ssd", "replicas", "ladder_ver",
                 "last_arr", "gaps", "lru", "future", "fptr", "pending", "hist_keep")

    def __init__(self, m, s):
        self.m, self.s = m, s
        self.has_dram = False
        self.has_ssd = False
        self.t_dram = 0.0
        self.t_ssd = 0.0
        self.replicas: List[Replica] = []
        self.ladder_ver = 0
        self.last_arr = None
        self.gaps = deque(maxlen=32)
        self.lru = 0.0
        self.future = None
        self.fptr = 0
        self.pending = deque()
        self.hist_keep = None


class Site:
    def __init__(self, idx, spec: SiteSpec, carbon: CarbonSeries, p_ctx_scale: float = 1.0):
        self.idx, self.spec, self.carbon = idx, spec, carbon
        base = GPU_TYPES[spec.gpu_type]
        self.gt = base if p_ctx_scale == 1.0 else replace(base, p_ctx_W=base.p_ctx_W * p_ctx_scale)
        self.gpus = [GPU(i, idx, self.gt) for i in range(spec.n_gpus)]
        self.dram_used = 0.0
        self.ssd_used = 0.0
        self.is_edge = spec.kind == "edge"


@dataclass
class Totals:
    E: Dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CATS})
    G: Dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CATS})


class Simulator:
    def __init__(self, sp: SimParams, wl: Workload, policy, sites: Optional[List[SiteSpec]] = None):
        self.sp, self.wl, self.policy = sp, wl, policy
        self.en = sp.energy
        specs = sites if sites is not None else default_sites(sp.edge_gpus)
        if sp.dram_scale != 1.0:
            specs = [replace(x, dram_GB=x.dram_GB * sp.dram_scale) for x in specs]
        hours = int(math.ceil((sp.duration_s + 2 * 86400) / 3600.0))
        self.carbon, self.carbon_sources = build_carbon(CARBON_SCENARIOS[sp.scenario], hours, sp.data_dir,
                                                        sp.carbon_forecast_noise, sp.seed,
                                                        sp.carbon_offset_h)
        self.sites = [Site(i, s, self.carbon[s.grid], sp.p_ctx_scale) for i, s in enumerate(specs)]
        self.site_by_name = {s.spec.name: s.idx for s in self.sites}
        rtt = default_rtt()
        self.rtt = np.array([[rtt.get((r, s.spec.name), 0.14) for s in self.sites] for r in USER_REGIONS])
        self.home = [self.site_by_name.get(HOME_EDGE[r], -1) for r in USER_REGIONS]
        self.models = wl.models
        M, S = len(self.models), len(self.sites)
        self.ms = [[ModelSite(m, s) for s in range(S)] for m in range(M)]
        self.fits = np.array([[self.models[m].size_GB * 1.2 + 2.0 <= self.sites[s].gt.mem_GB
                               for s in range(S)] for m in range(M)])
        self.rng = np.random.default_rng(sp.seed + 99)
        # request records
        n = len(wl)
        self.r_site = np.full(n, -1, int)
        self.r_disp = np.full(n, np.nan)
        self.r_start = np.full(n, np.nan)
        self.r_ttft = np.full(n, np.nan)
        self.r_done = np.full(n, np.nan)
        self.r_cold = np.zeros(n, int)          # 0 warm, 1 triggered load, 2 waited on a loading replica
        self.r_src = np.zeros(n, int)           # source tier of the load it waited for
        self.r_dyn_g = np.zeros(n)
        self.r_b = np.zeros(n, int)
        self.r_rep = [None] * n
        # state
        self.t = 0.0
        self.ev = []
        self.seq = 0
        self.rid = 0
        self.tot = Totals()
        self.G_site = np.zeros(len(self.sites))   # grams per site (for the full-boundary metric)
        self.G_on = np.zeros(len(self.sites))     # bare-idle + host share of powered GPUs (power-managed boundary)
        self.G_on_bare = np.zeros(len(self.sites))  # bare idle of powered GPUs only (hosts stay on)
        self.acc_end = sp.duration_s
        self.H = int(math.ceil(sp.duration_s / 3600.0)) + 1
        self.ts_G = {c: np.zeros(self.H) for c in CATS} if sp.record_timeseries else None
        self.samples = [] if sp.record_timeseries else None
        self.n_loads = {TIER_DRAM: 0, TIER_SSD: 0, TIER_REMOTE: 0}
        self.n_prewarm = 0
        self.decision_ns = 0
        self.n_decisions = 0
        self.replicas: Dict[int, Replica] = {}
        # checkpoints are pre-staged on every site's SSD where they fit
        for s in self.sites:
            for m in range(M):
                if self.fits[m][s.idx] and s.ssd_used + self.models[m].size_GB <= s.spec.ssd_GB:
                    self.set_ssd(self.ms[m][s.idx], True, 0.0)

    # ------------------------------------------------------------------ events
    def push(self, t, kind, payload):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, kind, payload))

    def call_at(self, t, fn, *args):
        if t > self.sp.duration_s:
            return
        self.push(max(t, self.t), EV_CALL, (fn, args))

    # -------------------------------------------------------------- accounting
    def add(self, cat, E, g, t1, t2, site=-1):
        w0, w1 = self.sp.warmup_s, self.acc_end
        if t2 <= w0 or t1 >= w1:
            return
        if t2 > t1 and (t1 < w0 or t2 > w1):
            frac = (min(t2, w1) - max(t1, w0)) / (t2 - t1)
            E, g, t1, t2 = E * frac, g * frac, max(t1, w0), min(t2, w1)
        self.tot.E[cat] += E
        self.tot.G[cat] += g
        if site >= 0:
            self.G_site[site] += g
        if self.ts_G is not None:
            if t2 - t1 < 1e-9:
                self.ts_G[cat][min(int(t1 // 3600), self.H - 1)] += g
            else:
                h, t = int(t1 // 3600), t1
                while t < t2 - 1e-9 and h < self.H:
                    e = min(t2, (h + 1) * 3600.0)
                    self.ts_G[cat][h] += g * (e - t) / (t2 - t1)
                    t, h = e, h + 1

    def gpu_acc(self, g: GPU, t):
        if t > g.last and g.residents:
            cat = "load" if g.n_loading else ("serve" if g.n_active else "idle")
            site = self.sites[g.site]
            self.add(cat, g.gt.p_ctx_W * (t - g.last), site.carbon.grams_power(g.gt.p_ctx_W, g.last, t), g.last, t,
                     g.site)
            # in a power-managed fleet a GPU (and its share of the host) is powered only while it holds a model
            w0, w1 = self.sp.warmup_s, self.acc_end
            a0, a1 = max(g.last, w0), min(t, w1)
            if a1 > a0:
                p_on = BARE_IDLE_W[site.spec.gpu_type] + HOST_BASE_W[site.spec.kind] / max(site.spec.n_gpus, 1)
                self.G_on[g.site] += site.carbon.grams_power(p_on, a0, a1)
                self.G_on_bare[g.site] += site.carbon.grams_power(BARE_IDLE_W[site.spec.gpu_type], a0, a1)
        g.last = max(g.last, t)

    def set_dram(self, ms: ModelSite, val: bool, t):
        if ms.has_dram == val:
            return
        size = self.models[ms.m].size_GB
        site = self.sites[ms.s]
        if val:
            ms.has_dram, ms.t_dram = True, t
            site.dram_used += size
        else:
            p = self.en.p_dram_W_per_GB * size
            self.add("dram", p * (t - ms.t_dram), site.carbon.grams_power(p, ms.t_dram, t), ms.t_dram, t, ms.s)
            ms.has_dram = False
            site.dram_used -= size

    def set_ssd(self, ms: ModelSite, val: bool, t):
        if ms.has_ssd == val:
            return
        size = self.models[ms.m].size_GB
        site = self.sites[ms.s]
        if val:
            ms.has_ssd, ms.t_ssd = True, t
            site.ssd_used += size
        else:
            p = self.en.p_ssd_W_per_GB * size
            self.add("ssd", p * (t - ms.t_ssd), site.carbon.grams_power(p, ms.t_ssd, t), ms.t_ssd, t, ms.s)
            ms.has_ssd = False
            site.ssd_used -= size

    # ------------------------------------------------------------ performance
    def speed(self, m):
        return self.models[m].params_B / 7.0

    def prefill_time(self, i, s):
        gt = self.sites[s].gt
        return self.wl.tin[i] / (gt.prefill_tps_7b / self.speed(self.wl.model[i]))

    def step_time(self, m, s, b):
        gt = self.sites[s].gt
        return gt.step_s_7b * self.speed(m) * (1.0 + gt.step_slope * (max(b, 1) - 1))

    def p_dyn(self, s, b):
        gt = self.sites[s].gt
        return gt.p_dyn_max_W * min(1.0, 0.3 + 0.7 * b / max(gt.max_batch / 2.0, 1.0))

    def service(self, i, s, b):
        """(prefill_s, decode_s, dynamic_energy_J, attributable_baseline_J) for request i on site s."""
        m = self.wl.model[i]
        gt = self.sites[s].gt
        pf = self.prefill_time(i, s)
        st = self.step_time(m, s, b)
        dec = self.wl.tout[i] * st
        e = pf * gt.p_dyn_max_W * self.en.prefill_power_frac + self.wl.tout[i] * st * self.p_dyn(s, b) / max(b, 1)
        base = gt.p_ctx_W * (pf + dec) / max(b, 1)
        return pf, dec, e, base

    def load_time(self, m, s, tier):
        sp = self.sites[s].spec
        size = self.models[m].size_GB
        bw = {TIER_DRAM: sp.bw_dram_GBs, TIER_SSD: sp.bw_ssd_GBs, TIER_REMOTE: sp.bw_remote_GBs}[tier]
        return self.en.init_overhead_s + size / bw

    def load_energy(self, m, s, tier):
        if tier == TIER_GPU:
            return 0.0
        return (self.sites[s].gt.p_ctx_W + self.en.p_load_extra_W) * self.load_time(m, s, tier)

    def best_tier(self, ms: ModelSite):
        return TIER_DRAM if ms.has_dram else (TIER_SSD if ms.has_ssd else TIER_REMOTE)

    def need_GB(self, m):
        return self.models[m].size_GB * 1.2 + 2.0

    # ---------------------------------------------------------- site queries
    def ready_spare(self, ms: ModelSite):
        best = None
        for r in ms.replicas:
            if r.state == "ready" and r.n_active < self.sites[ms.s].gt.max_batch:
                if best is None or r.n_active < best.n_active:
                    best = r
        return best

    def can_host(self, m, s):
        if not self.fits[m][s]:
            return False
        if self.ms[m][s].replicas:
            return True
        need = self.need_GB(m)
        site = self.sites[s]
        if any(g.mem_free >= need for g in site.gpus):
            return True
        for g in site.gpus:  # free memory obtainable by evicting idle replicas on this GPU
            free = g.mem_free
            for rid in g.residents:
                r = self.replicas[rid]
                if r.state == "ready" and r.n_active == 0 and not r.queue:
                    free += r.need
            if free >= need:
                return True
        return False

    def startup_estimate(self, m, s):
        """Seconds until a replica of m at s could start a new request."""
        ms = self.ms[m][s]
        if self.ready_spare(ms) is not None:
            return 0.0
        loading = [r.ready_t for r in ms.replicas if r.state == "loading"]
        if loading:
            return max(min(loading) - self.t, 0.0)
        if ms.replicas:  # all full: rough queueing estimate
            r = min(ms.replicas, key=lambda x: len(x.queue))
            return (len(r.queue) + 1) * 2.0
        return self.load_time(m, s, self.best_tier(ms))

    def est_ttft(self, i, s):
        reg = self.wl.region[i]
        return self.rtt[reg][s] + self.startup_estimate(self.wl.model[i], s) + self.prefill_time(i, s)

    def marginal_carbon(self, i, s, intensity=None, include_load=True):
        """Estimated grams for serving i at s now (serving + a cold load if one is needed)."""
        m = self.wl.model[i]
        ms = self.ms[m][s]
        rep = self.ready_spare(ms)
        b = (rep.n_active + 1) if rep is not None else 1
        pf, dec, e, base = self.service(i, s, b)
        E = e + base
        if include_load and rep is None and not any(r.state == "loading" for r in ms.replicas):
            E += self.load_energy(m, s, self.best_tier(ms))
        I = self.sites[s].carbon.I(self.t) if intensity is None else intensity
        return E * I / KWH_J

    def predicted_gap(self, ms: ModelSite):
        if len(ms.gaps) < 3:
            return None
        g = float(np.median(ms.gaps))
        if self.sp.prediction_noise > 0:
            g *= float(self.rng.lognormal(0.0, self.sp.prediction_noise))
        return g

    # ------------------------------------------------------------ mechanics
    def alloc_gpu(self, m, s, t) -> Optional[GPU]:
        need = self.need_GB(m)
        site = self.sites[s]
        fit = [g for g in site.gpus if g.mem_free >= need]
        if fit:  # pack onto already-active GPUs first (one CUDA context is paid per GPU)
            return min(fit, key=lambda g: (0 if g.residents else 1, g.mem_free))
        idle = []
        for g in site.gpus:
            for rid in g.residents:
                r = self.replicas[rid]
                if r.state == "ready" and r.n_active == 0 and not r.queue:
                    idle.append(r)
        idle.sort(key=lambda r: r.last_used)
        for r in idle:
            self.demote_replica(r, t, reason="evict", retry=False)  # no re-entrant retry
            fit = [g for g in site.gpus if g.mem_free >= need]
            if fit:
                return min(fit, key=lambda g: g.mem_free)
        return None

    def start_load(self, m, s, t, prewarm=False) -> Optional[Replica]:
        gpu = self.alloc_gpu(m, s, t)
        if gpu is None:
            return None
        ms = self.ms[m][s]
        tier = self.best_tier(ms)
        lt = self.load_time(m, s, tier)
        self.gpu_acc(gpu, t)
        self.rid += 1
        r = Replica(self.rid, m, s, gpu, self.need_GB(m))
        r.src = tier
        r.prewarmed = prewarm
        r.ready_t = t + lt
        self.replicas[r.id] = r
        gpu.residents.add(r.id)
        gpu.n_loading += 1
        gpu.mem_free -= r.need
        ms.replicas.append(r)
        ms.ladder_ver += 1  # cancels cache-ladder timers
        ms.lru = t
        if tier == TIER_REMOTE:
            self.ensure_ssd(ms, t)
        self.n_loads[tier] += 1
        if prewarm:
            self.n_prewarm += 1
        site = self.sites[s]
        ex = self.en.p_load_extra_W
        self.add("load", ex * lt, site.carbon.grams_power(ex, t, t + lt), t, t + lt, s)
        self.push(t + lt, EV_LOADED, r.id)
        return r

    def ensure_ssd(self, ms: ModelSite, t):
        site = self.sites[ms.s]
        size = self.models[ms.m].size_GB
        if ms.has_ssd:
            return
        if site.ssd_used + size > site.spec.ssd_GB:
            victims = sorted((x for x in (self.ms[mm][ms.s] for mm in range(len(self.models)))
                              if x.has_ssd and not x.replicas and not x.has_dram), key=lambda x: x.lru)
            for v in victims:
                self.set_ssd(v, False, t)
                if site.ssd_used + size <= site.spec.ssd_GB:
                    break
        self.set_ssd(ms, True, t)

    def ensure_dram(self, ms: ModelSite, t) -> bool:
        site = self.sites[ms.s]
        size = self.models[ms.m].size_GB
        if ms.has_dram:
            return True
        if size > site.spec.dram_GB:
            return False
        if site.dram_used + size > site.spec.dram_GB:
            victims = sorted((x for x in (self.ms[mm][ms.s] for mm in range(len(self.models)))
                              if x.has_dram and x is not ms), key=lambda x: x.lru)
            for v in victims:
                self.set_dram(v, False, t)
                if site.dram_used + size <= site.spec.dram_GB:
                    break
        if site.dram_used + size > site.spec.dram_GB:
            return False
        self.set_dram(ms, True, t)
        return True

    def demote_replica(self, r: Replica, t, to=None, reason="timer", retry=True):
        """Remove replica r from its GPU; weights go to `to` ('dram'|'ssd'|'remote')."""
        if r.id not in self.replicas or r.state != "ready" or r.n_active or r.queue:
            return False
        to = to or self.policy.gpu_demote_to
        g = r.gpu
        self.gpu_acc(g, t)
        g.residents.discard(r.id)
        g.mem_free += r.need
        del self.replicas[r.id]
        ms = self.ms[r.m][r.s]
        ms.replicas.remove(r)
        ms.lru = t
        if to == "dram":
            if not self.ensure_dram(ms, t):
                self.ensure_ssd(ms, t)
        elif to == "ssd":
            self.ensure_ssd(ms, t)
        elif to == "remote":
            if not ms.replicas:
                self.set_dram(ms, False, t)
                self.set_ssd(ms, False, t)
        if not ms.replicas:
            self.policy.on_offgpu(self, ms, t)
        if retry:
            self.retry_pending(r.s, t)
        return True

    def demote_cache(self, ms: ModelSite, t, level):
        """level: 'dram->ssd' or 'ssd->remote'."""
        if ms.replicas:
            return
        if level == "dram->ssd" and ms.has_dram:
            self.ensure_ssd(ms, t)
            self.set_dram(ms, False, t)
        elif level == "ssd->remote" and ms.has_ssd and not ms.has_dram:
            self.set_ssd(ms, False, t)

    def retry_pending(self, s, t):
        if getattr(self, "_retrying", False):   # guard against re-entrance under GPU pressure
            return
        self._retrying = True
        try:
            self._retry_pending(s, t)
        finally:
            self._retrying = False

    def _retry_pending(self, s, t):
        for m in range(len(self.models)):
            ms = self.ms[m][s]
            while ms.pending:
                i = ms.pending[0]
                if not self.try_enqueue(i, s, t, from_pending=True):
                    break
                ms.pending.popleft()

    # --------------------------------------------------------------- requests
    def dispatch(self, i, s, t):
        """Send request i to site s at time t."""
        m = self.wl.model[i]
        ms = self.ms[m][s]
        self.r_site[i] = s
        self.r_disp[i] = t
        if ms.last_arr is not None and t - ms.last_arr > 1.0:
            ms.gaps.append(t - ms.last_arr)
        ms.last_arr = t
        ms.lru = t
        if not self.try_enqueue(i, s, t):
            # no GPU memory at s: try other sites by RTT, else wait at s
            reg = self.wl.region[i]
            for s2 in np.argsort(self.rtt[reg]):
                s2 = int(s2)
                if s2 != s and self.fits[m][s2] and self.can_host(m, s2):
                    self.r_site[i] = s2
                    if self.try_enqueue(i, s2, t):
                        return
            self.r_site[i] = s
            ms.pending.append(i)

    def try_enqueue(self, i, s, t, from_pending=False) -> bool:
        m = self.wl.model[i]
        ms = self.ms[m][s]
        rep = self.ready_spare(ms)
        if rep is not None:
            self.start_request(i, rep, t)
            return True
        loading = [r for r in ms.replicas if r.state == "loading"]
        if loading:
            r = min(loading, key=lambda x: len(x.queue))
            r.queue.append(i)
            self.r_cold[i] = 2
            self.r_src[i] = r.src
            return True
        r = self.start_load(m, s, t)
        if r is not None:
            r.queue.append(i)
            self.r_cold[i] = 1
            self.r_src[i] = r.src
            return True
        if ms.replicas:  # all replicas busy: queue on the least loaded
            r = min(ms.replicas, key=lambda x: len(x.queue))
            r.queue.append(i)
            return True
        return False

    def start_request(self, i, r: Replica, t):
        r.prewarmed = False
        if r.idle_since is not None:
            r.idle_since = None
            r.ver += 1  # cancel idle timers
        g = r.gpu
        self.gpu_acc(g, t)
        r.n_active += 1
        g.n_active += 1
        b = r.n_active
        s = r.s
        pf, dec, e, base = self.service(i, s, b)
        done = t + pf + dec
        reg = self.wl.region[i]
        self.r_start[i] = t
        self.r_ttft[i] = (t - self.wl.t[i]) + self.rtt[reg][s] + pf
        if self.wl.cls[i] == 0 or not self.policy.defers:
            self.policy.observe(self, self.r_ttft[i] <= self.sp.slo_ttft_s)
        self.r_b[i] = b
        self.r_rep[i] = r.id
        site = self.sites[s]
        gdyn = site.carbon.grams_energy(e, t, done)
        self.r_dyn_g[i] = gdyn
        self.add("serve", e, gdyn, t, done, s)
        r.last_used = t
        self.push(done, EV_DONE, (i, r.id))

    def on_done(self, i, rid, t):
        self.r_done[i] = t + self.rtt[self.wl.region[i]][self.r_site[i]] * 0.5
        r = self.replicas.get(rid)
        if r is None:
            return
        g = r.gpu
        self.gpu_acc(g, t)
        r.n_active -= 1
        g.n_active -= 1
        r.last_used = t
        B = self.sites[r.s].gt.max_batch
        while r.queue and r.n_active < B:
            self.start_request(r.queue.popleft(), r, t)
        if r.n_active == 0 and not r.queue:
            self.replica_idle(r, t)

    def on_loaded(self, rid, t):
        r = self.replicas.get(rid)
        if r is None:
            return
        g = r.gpu
        self.gpu_acc(g, t)
        g.n_loading -= 1
        r.state = "ready"
        r.ready_t = t
        r.last_used = t
        B = self.sites[r.s].gt.max_batch
        while r.queue and r.n_active < B:
            self.start_request(r.queue.popleft(), r, t)
        # spill: queued requests of the same model on busy replicas may move here
        if r.n_active < B:
            for o in self.ms[r.m][r.s].replicas:
                while o is not r and o.queue and o.state == "ready" and r.n_active < B:
                    self.start_request(o.queue.popleft(), r, t)
        if r.n_active == 0 and not r.queue:
            self.replica_idle(r, t)

    def replica_idle(self, r: Replica, t):
        r.idle_since = t
        r.ver += 1
        t0 = time.perf_counter_ns()
        self.policy.on_replica_idle(self, r, t)
        self.decision_ns += time.perf_counter_ns() - t0
        self.n_decisions += 1

    def prewarm(self, m, s, t, keep_until=None):
        ms = self.ms[m][s]
        if ms.replicas or not self.fits[m][s]:
            return None
        r = self.start_load(m, s, t, prewarm=True)
        if r is not None:
            r.keep_until = keep_until
        return r

    # ------------------------------------------------------------------- run
    def run(self):
        wl = self.wl
        self.policy.init(self)
        for i in range(len(wl)):
            self.push(float(wl.t[i]), EV_ARR, i)
        if self.samples is not None:
            self.push(0.0, EV_SAMPLE, None)
        T = self.sp.duration_s
        wall0 = time.perf_counter()
        while self.ev:
            t, _, kind, p = heapq.heappop(self.ev)
            self.t = t
            if kind == EV_ARR:
                i = p
                if wl.cls[i] == 1 and self.policy.defers:
                    t0 = time.perf_counter_ns()
                    self.policy.on_deferrable(self, i, t)
                    self.decision_ns += time.perf_counter_ns() - t0
                    self.n_decisions += 1
                else:
                    t0 = time.perf_counter_ns()
                    s = self.policy.place(self, i, t)
                    self.decision_ns += time.perf_counter_ns() - t0
                    self.n_decisions += 1
                    self.dispatch(i, s, t)
            elif kind == EV_DONE:
                self.on_done(p[0], p[1], t)
            elif kind == EV_LOADED:
                self.on_loaded(p, t)
            elif kind == EV_CALL:
                fn, args = p
                t0 = time.perf_counter_ns()
                fn(self, t, *args)
                self.decision_ns += time.perf_counter_ns() - t0
            elif kind == EV_SAMPLE:
                self.sample(t)
                if t + self.sp.sample_period_s <= T:
                    self.push(t + self.sp.sample_period_s, EV_SAMPLE, None)
        self.finalize(T)
        self.wall_s = time.perf_counter() - wall0
        return self

    def sample(self, t):
        gpu_gb = sum(r.need for r in self.replicas.values())
        dram = sum(s.dram_used for s in self.sites)
        ssd = sum(s.ssd_used for s in self.sites)
        n_rep = len(self.replicas)
        busy = sum(1 for r in self.replicas.values() if r.n_active)
        self.samples.append((t, gpu_gb, dram, ssd, n_rep, busy))

    def finalize(self, T):
        for s in self.sites:
            for g in s.gpus:
                if g.last < T:
                    self.gpu_acc(g, T)
        for row in self.ms:
            for ms in row:
                if ms.has_dram:
                    self.set_dram(ms, False, T)
                if ms.has_ssd:
                    self.set_ssd(ms, False, T)
