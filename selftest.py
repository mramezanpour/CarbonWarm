#!/usr/bin/env python3
"""Correctness checks for the simulator. Run:  python selftest.py"""
import math
import sys
import warnings

import numpy as np

from carbonwarm.carbon import KWH_J, CarbonSeries
from carbonwarm.config import SimParams, default_sites, model_catalog
from carbonwarm.metrics import run_metrics
from carbonwarm.policies import ABLATIONS, ORACLE, POLICIES, make_policy
from carbonwarm.sim import TIER_DRAM, Simulator
from carbonwarm.workload import Workload, build_workload

warnings.simplefilter("ignore")
FAIL = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def one_request_workload(sp, model=0, region=0, t=10.0):
    return Workload(t=np.array([t]), model=np.array([model]), tin=np.array([500]), tout=np.array([200]),
                    region=np.array([region]), cls=np.array([0]), deadline=np.array([np.inf]),
                    models=model_catalog(sp.n_models), source="unit")


def constant_carbon(sim, value=100.0):
    hours = len(next(iter(sim.carbon.values())).values)
    for s in sim.sites:
        s.carbon = CarbonSeries("const", np.full(hours, value))


# 1. carbon integrals --------------------------------------------------------
c = CarbonSeries("t", np.array([100.0, 200.0, 300.0]))
check("integral across hour boundary", abs(c.integral(1800, 5400) - (100 * 1800 + 200 * 1800)) < 1e-6)
check("grams of 1 kW for 1 h at 100 g/kWh = 100 g", abs(c.grams_power(1000, 0, 3600) - 100.0) < 1e-9)
check("best_start picks the greenest hour", c.best_start(0, 3 * 3600 - 1)[0] == 0.0)

# 2. single request with Fixed-KA: exact energy bookkeeping ------------------
sp = SimParams(duration_s=3600, n_models=4, seed=0)
wl = one_request_workload(sp)
sim = Simulator(sp, wl, make_policy("Fixed-KA"))
constant_carbon(sim, 100.0)
sim.run()
s = sim.sites[sim.r_site[0]]
lt = sim.load_time(0, s.idx, 2)            # loaded from local SSD
pf, dec, e_dyn, _ = sim.service(0, s.idx, 1)
E = sim.tot.E
check("load energy = (p_ctx + p_extra) * load time",
      abs(E["load"] - (s.gt.p_ctx_W + sp.energy.p_load_extra_W) * lt) < 1e-6, f"({E['load']:.1f} J)")
check("idle energy = p_ctx * 600 s keep-alive", abs(E["idle"] - s.gt.p_ctx_W * 600.0) < 1e-6, f"({E['idle']:.1f} J)")
check("serve energy = p_ctx*(prefill+decode) + dynamic",
      abs(E["serve"] - (s.gt.p_ctx_W * (pf + dec) + e_dyn)) < 1e-6)
check("carbon = energy x 100 g/kWh", abs(sum(sim.tot.G.values()) - sum(E.values()) * 100 / KWH_J) < 1e-9)
check("TTFT = load + rtt + prefill", abs(sim.r_ttft[0] - (lt + sim.rtt[0][s.idx] + pf)) < 1e-9)

# 3. CarbonWarm ladder: GPU -> DRAM step happens exactly at the ski-rental threshold
sp = SimParams(duration_s=6 * 3600, n_models=4, seed=0, green_prewarm=False)
wl = one_request_workload(sp)
pol = make_policy("CW w/o prediction")
sim = Simulator(sp, wl, pol)
constant_carbon(sim, 100.0)
sim.run()
si = sim.r_site[0]
site = sim.sites[si]
size = sim.models[0].size_GB
p_hi = site.gt.p_ctx_W
p_lo = sp.energy.p_dram_W_per_GB * size
thr = sim.load_energy(0, si, TIER_DRAM) * 100 / KWH_J + sp.mu_g_per_violation * pol.violates(sim, 0, si, TIER_DRAM)
T = thr / ((p_hi - p_lo) * 100 / KWH_J)
check("ladder demotes after T* = threshold / net holding rate",
      abs(sim.tot.E["idle"] - p_hi * T) < 1e-3, f"(T*={T:.1f}s, idle E={sim.tot.E['idle']:.1f} J)")

# 4. all policies run, determinism, nothing lost ---------------------------------
sp = SimParams(duration_s=3 * 3600, n_models=6, target_rate_rps=0.3, seed=3)
wl = build_workload(sp)
ref = None
for name in list(POLICIES) + list(ABLATIONS):
    m = run_metrics(Simulator(sp, wl, make_policy(name)).run())
    check(f"{name}: all requests served", m["n_unserved"] == 0)
    check(f"{name}: carbon > 0 and finite", m["carbon_total_kg"] > 0 and math.isfinite(m["carbon_total_kg"]))
    if name == "CarbonWarm":
        ref = m
m2 = run_metrics(Simulator(sp, wl, make_policy("CarbonWarm")).run())
check("deterministic for a fixed seed", m2["carbon_total_kg"] == ref["carbon_total_kg"])
check("CarbonWarm meets deferrable deadlines", ref["deadline_miss_pct"] < 1.0, f"({ref['deadline_miss_pct']:.2f}%)")

# 5. clairvoyant baseline replays CarbonWarm's placement and serves every request -----------
cw = Simulator(sp, wl, make_policy("CarbonWarm")).run()
replay = {"site": cw.r_site.copy(), "disp": cw.r_disp.copy()}
orc = Simulator(sp, wl, make_policy(ORACLE, replay=replay)).run()
mo = run_metrics(orc)
res = lambda x: x.tot.G["idle"] + x.tot.G["load"] + x.tot.G["dram"] + x.tot.G["ssd"]
check("clairvoyant baseline: all requests served, finite carbon",
      mo["n_unserved"] == 0 and math.isfinite(mo["carbon_total_kg"]),
      f"(residency carbon: clairvoyant {res(orc):.1f} g, CarbonWarm {res(cw):.1f} g; not a lower bound)")

# 6. SLO controller: probabilistic violation estimate and safety margin -------------------
pol = make_policy("CarbonWarm")
sim = Simulator(sp, wl, pol).run()
v = [pol.violates(sim, m, s, t) for m in range(len(sim.models)) for s in range(len(sim.sites))
     for t in (1, 2, 3) if sim.fits[m][s]]
check("violation estimate is a probability in [0, 1]", all(0.0 <= x <= 1.0 for x in v),
      f"(distinct values: {len(set(round(x, 3) for x in v))})")
m6 = run_metrics(sim)
check("CarbonWarm meets the SLO target on the test workload", m6["slo_attain_pct"] >= sp.slo_target_pct,
      f"({m6['slo_attain_pct']:.2f}% vs target {sp.slo_target_pct:g}%)")

# 7. long-tail catalogue, Zipf permutation -------------------------------------------
cat = model_catalog(200)
check("catalogue extends to 200 models with fine-tuned variants",
      len(cat) == 200 and cat[16].name.endswith("-ft1") and cat[16].size_GB == cat[0].size_GB)
sp7 = SimParams(duration_s=3 * 3600, n_models=6, target_rate_rps=0.3, seed=3)
w0 = build_workload(sp7)
w1 = build_workload(sp7.copy(zipf_perm=1))
c0, c1 = np.bincount(w0.model, minlength=6), np.bincount(w1.model, minlength=6)
check("Zipf permutation moves popularity between models", sorted(c0) == sorted(c1) and not np.array_equal(c0, c1))

# 8. per-site carbon, full boundary, new baselines ------------------------------------
fb = {}
for name in ["Carbon-Routing+CS", "Carbon-Routing+DRAM", "MultiState-PD", "CarbonWarm", "Fixed-KA"]:
    sm = Simulator(sp7, w0, make_policy(name)).run()
    mm = run_metrics(sm)
    check(f"{name} serves every request", mm["n_unserved"] == 0)
    check(f"{name} per-site carbon sums to the total",
          abs(mm["carbon_site_sum_kg"] - mm["carbon_total_kg"]) < 1e-9 * max(1.0, mm["carbon_total_kg"]) + 1e-12)
    check(f"{name} full-boundary carbon exceeds the policy carbon", mm["carbon_full_kg"] > mm["carbon_total_kg"])
    fb[name] = mm["carbon_full_base_kg"]
check("full-boundary baseline is identical for all policies", max(fb.values()) - min(fb.values()) < 1e-9)

# 9. multi-state power-down demotes at the lower-envelope crossing ---------------------
sp9 = SimParams(duration_s=6 * 3600, n_models=4, seed=0)
wl9 = one_request_workload(sp9)
sim9 = Simulator(sp9, wl9, make_policy("MultiState-PD"))
constant_carbon(sim9, 100.0)
sim9.run()
s9 = sim9.r_site[0]
size9 = sim9.models[0].size_GB
hG = sim9.sites[s9].gt.p_ctx_W
cands9 = [(sim9.load_energy(0, s9, TIER_DRAM), sp9.energy.p_dram_W_per_GB * size9)]
x1 = min(e / (hG - h) for e, h in cands9 + [(sim9.load_energy(0, s9, 2), sp9.energy.p_ssd_W_per_GB * size9),
                                            (sim9.load_energy(0, s9, 3), 0.0)])
check("MultiState-PD leaves the GPU at the first lower-envelope crossing",
      abs(sim9.tot.E["idle"] - hG * x1) < 1e-3, f"(x1={x1:.1f}s)")

# 10. context-power scaling and DRAM scaling -------------------------------------------
sp10 = SimParams(duration_s=3600, n_models=4, seed=0)
wl10 = one_request_workload(sp10)
e = []
for sc in (1.0, 0.5):
    sim10 = Simulator(sp10.copy(p_ctx_scale=sc), wl10, make_policy("Fixed-KA"))
    constant_carbon(sim10, 100.0)
    sim10.run()
    e.append(sim10.tot.E["idle"])
check("p_ctx_scale scales the idle energy", abs(e[1] - 0.5 * e[0]) < 1e-6)
sim10 = Simulator(sp10.copy(dram_scale=0.25), wl10, make_policy("Fixed-KA"))
check("dram_scale shrinks the DRAM cache", all(abs(s.spec.dram_GB - 0.25 * d.dram_GB) < 1e-9
                                                    for s, d in zip(sim10.sites, default_sites(sp10.edge_gpus))))

# 11. the SLO price is recorded ------------------------------------------------------
pol11 = make_policy("CarbonWarm")
Simulator(sp7, w0, pol11).run()
check("mu trajectory is recorded", len(pol11.mu_trace) > 0 and all(1e-3 <= x[1] <= 1e3 for x in pol11.mu_trace))

from carbonwarm.metrics import full_boundary
sim_pm = Simulator(sp7, w0, make_policy("Fixed-KA")); sim_pm.run(); fb = full_boundary(sim_pm)
check("power-managed boundary lies between policy carbon and full boundary",
      sim_pm.G_site.sum() / 1000.0 <= fb["carbon_full_pm_kg"] <= fb["carbon_full_kg"] + 1e-9,
      f"({fb['carbon_full_pm_kg']:.3f} kg)")
check("GPU-only power management lies between the power-managed and the full boundary",
      fb["carbon_full_pm_kg"] - 1e-9 <= fb["carbon_full_pmgpu_kg"] <= fb["carbon_full_kg"] + 1e-9,
      f"({fb['carbon_full_pmgpu_kg']:.3f} kg)")

print("\nALL TESTS PASSED" if not FAIL else f"\n{len(FAIL)} TEST(S) FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
