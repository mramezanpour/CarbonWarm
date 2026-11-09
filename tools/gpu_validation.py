#!/usr/bin/env python3
"""Measure an NVIDIA GPU to validate the power and timing model of the simulator.

Measures bare idle power, idle power with a CUDA context and with a resident model, checkpoint loads from SSD
and pinned host memory, prefill throughput and decode energy per token at batch sizes 1, 8 and 32, using the
NVML energy counter. Writes results_gpu/gpu_validation.json and .csv.

Usage: python tools/gpu_validation.py      (set CW_COLAB=1 on machines that cannot be rebooted)
The first run only downloads the models; reboot before the measurement run so that SSD reads are not cached.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time

# ----------------------------------------------------------------------------- settings
OUT_DIR = "results_gpu"
CACHE_DIR = "gpu_models"           # models are downloaded here once (a few GB)
IDLE_SECONDS = 60                  # length of each idle measurement
SAMPLE_HZ = 20                     # NVML sampling rate
BATCHES = [1, 8, 32]               # decode batch sizes (reduced automatically if memory is short)
NEW_TOKENS = 128                   # generated tokens per decode measurement
PREFILL_TOKENS = 1000
MIN_ENERGY_WINDOW_S = 1.0         # NVML energy is only trusted for measurement windows at least this long
# Open models without a licence gate; chosen automatically by GPU memory (fp16 needs ~2 bytes/parameter)
MODELS = [  # (Hugging Face id, parameters in billions)
    ("Qwen/Qwen2.5-0.5B-Instruct", 0.5),
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    ("Qwen/Qwen2.5-3B-Instruct", 3.1),
    ("Qwen/Qwen2.5-7B-Instruct", 7.6),
]
# ----------------------------------------------------------------------------- end of settings


os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
         "F8_E4M3": 1, "F8_E5M2": 1}
INFERENCE_OK = True
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the carbonwarm_sim folder


def need(pkg, pip_name=None):
    try:
        return __import__(pkg)
    except ImportError:
        sys.exit(f"Missing package '{pkg}'. Install it with:  pip install {pip_name or pkg}")


torch = need("torch")
pynvml = need("pynvml", "nvidia-ml-py")
need("transformers")
need("safetensors")
need("huggingface_hub")
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def load_model(path):
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda")
    except TypeError:  # older transformers
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="cuda")


def inference_self_check():
    """transformers hides the real reason when a model class cannot be imported; find and print it."""
    try:
        from transformers import Qwen2ForCausalLM  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        pass
    try:
        import importlib
        importlib.import_module("transformers.models.qwen2.modeling_qwen2")
        return True
    except Exception as e:  # noqa: BLE001
        print("\n!! transformers cannot import the Qwen2 model class. Real reason:")
        print(f"   {type(e).__name__}: {e}")
        print("   Fix in Anaconda Prompt:  pip install -U transformers accelerate")
        print("   If the message mentions torchvision:  pip uninstall -y torchvision")
        if "PIL" in str(e) or "Resampling" in str(e):
            print("   This one is an old Pillow (Image.Resampling needs Pillow >= 9.1):  pip install -U pillow")
        print("   The power and loading measurements still run; only prefill/decode are skipped.\n")
        return False


class PowerMeter:
    """Samples GPU power in a background thread; energy from NVML's counter when available."""

    def __init__(self, handle, hz=SAMPLE_HZ):
        self.h, self.dt = handle, 1.0 / hz
        self.samples, self._run = [], False
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            self.counter = True
        except pynvml.NVMLError:
            self.counter = False

    def _loop(self):
        while self._run:
            self.samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0))
            time.sleep(self.dt)

    def __enter__(self):
        self.samples = []
        self.e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h) if self.counter else None
        self.t0 = time.perf_counter()
        self._run = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self.t1 = time.perf_counter()
        self._run = False
        self.th.join()
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h) if self.counter else None
        self.seconds = self.t1 - self.t0
        if self.counter:
            self.joules = (e1 - self.e0) / 1000.0
        else:  # trapezoid over samples
            s = self.samples
            self.joules = sum((s[k + 1][0] - s[k][0]) * (s[k + 1][1] + s[k][1]) / 2 for k in range(len(s) - 1))
        self.avg_w = self.joules / max(self.seconds, 1e-9)
        self.peak_w = max((p for _, p in self.samples), default=float("nan"))


def idle(meter, seconds, label):
    print(f"  measuring {label} for {seconds} s ... ", end="", flush=True)
    with meter as m:
        time.sleep(seconds)
    print(f"{m.avg_w:.1f} W")
    return m.avg_w


def folder_size_gb(paths):
    return sum(os.path.getsize(p) for p in paths) / 1e9


def main():
    global OUT_DIR
    OUT_DIR = os.path.join(BASE_DIR, OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(h)
    gpu_name = gpu_name.decode() if isinstance(gpu_name, bytes) else gpu_name
    vram_gb = pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**30
    try:
        tdp_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
    except pynvml.NVMLError:
        tdp_w = float("nan")
    print(f"GPU: {gpu_name}, {vram_gb:.1f} GB, power limit {tdp_w:.0f} W")
    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot see the GPU. Run tools/check_gpu.py.")
    meter = PowerMeter(h)
    print(f"Energy source: {'NVML energy counter' if meter.counter else 'integrated power samples'}")

    # models that fit (weights + KV cache + activations for batch 32)
    models = [(mid, p) for mid, p in MODELS if 2.0 * p * 1.25 + 1.5 <= vram_gb]
    if not models:
        sys.exit("The GPU memory is too small even for the smallest model.")
    models = models[-3:]  # up to three sizes
    print("Models:", ", ".join(m for m, _ in models))

    # 0. downloads (first run only). Freshly downloaded files sit in the operating system's file cache in RAM,
    # so an "SSD" load right after the download would really be a RAM load. Therefore: download, restart the
    # computer, run again.
    pats = ["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"]
    # Plain folders (no symbolic links): Windows refuses symlinks without administrator rights (WinError 1314).
    local, missing = {}, []
    for mid, _ in models:
        d = os.path.join(BASE_DIR, CACHE_DIR, mid.replace("/", "--"))
        local[mid] = d
        complete = os.path.isfile(os.path.join(d, "config.json")) and any(
            f.endswith(".safetensors") for f in (os.listdir(d) if os.path.isdir(d) else []))
        if not complete:
            missing.append(mid)
    if missing:
        for mid in missing:
            print(f"Downloading {mid} (only on the first run) ...")
            try:
                snapshot_download(mid, local_dir=local[mid], allow_patterns=pats, local_dir_use_symlinks=False)
            except TypeError:  # newer huggingface_hub: the argument no longer exists, files are copied anyway
                snapshot_download(mid, local_dir=local[mid], allow_patterns=pats)
        if os.environ.get("CW_COLAB") != "1":
            print("\nDownloads finished. RESTART THE COMPUTER, then run this script again for the measurements.")
            return
        # Cloud notebook (Google Colab): a reboot is not possible. Try to empty the file cache instead; if that is
        # not allowed, the SSD numbers of this run are cache-inflated and must not be used (all other numbers are fine).
        rc = os.system("sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null")
        print("\nDownloads finished (CW_COLAB=1): continuing without a restart; file cache "
              + ("dropped." if rc == 0 else "could NOT be dropped -> ignore the SSD results of this run."))
    global INFERENCE_OK
    INFERENCE_OK = inference_self_check()
    res = dict(gpu=gpu_name, vram_GB=vram_gb, power_limit_W=tdp_w, torch=torch.__version__,
               cuda=torch.version.cuda, energy_source="counter" if meter.counter else "samples", models={})

    # A. bare idle: no CUDA context created yet in this process
    print("\nA. Bare idle (no CUDA context). Do not touch the computer.")
    time.sleep(10)
    res["P_bare_W"] = idle(meter, IDLE_SECONDS, "bare idle")

    # B. CUDA context
    print("B. CUDA context created.")
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    time.sleep(5)
    res["P_ctx_W"] = idle(meter, IDLE_SECONDS, "idle with CUDA context")
    res["context_overhead_W"] = res["P_ctx_W"] - res["P_bare_W"]

    for mid, params in models:
        print(f"\n=== {mid} ({params} B parameters) ===")
        files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(local[mid]) for f in fs
                       if f.endswith(".safetensors"))
        size = folder_size_gb(files)
        r = dict(params_B=params, size_GB=size)

        # D1. SSD -> GPU in four DISJOINT groups of tensors (about 10/20/30/40 % of the checkpoint). Every byte is
        # read from the SSD exactly once after the reboot, so each group is a true SSD read, and we get four
        # (size, time, energy) points per model instead of one.
        torch.cuda.empty_cache()
        tensors = []
        for f in files:
            with safe_open(f, framework="pt", device="cpu") as fh:
                for k in fh.keys():
                    sl = fh.get_slice(k)
                    n = 1
                    for d in sl.get_shape():
                        n *= d
                    tensors.append((f, k, n * BYTES.get(sl.get_dtype(), 2)))
        total = sum(b for _, _, b in tensors)
        groups, acc, cut = [[]], 0, [0.1, 0.3, 0.6, 1.01]
        for t in tensors:
            if acc >= cut[len(groups) - 1] * total and len(groups) < 4:
                groups.append([])
            groups[-1].append(t)
            acc += t[2]
        sd, r["ssd_points"] = {}, []
        for g in groups:
            gb = sum(b for _, _, b in g) / 1e9
            with meter as m:
                for f in sorted({x[0] for x in g}):
                    keys = [x[1] for x in g if x[0] == f]
                    with safe_open(f, framework="pt", device="cuda") as fh:
                        for k in keys:
                            sd[k] = fh.get_tensor(k)
                torch.cuda.synchronize()
            r["ssd_points"].append(dict(size_GB=gb, s=m.seconds, J=m.joules, avg_W=m.avg_w))
            print(f"  SSD -> GPU  {gb:5.2f} GB: {m.seconds:.2f} s, {m.joules:.0f} J, {gb / m.seconds:.2f} GB/s")
        t_ssd = sum(x["s"] for x in r["ssd_points"])
        e_ssd = sum(x["J"] for x in r["ssd_points"])
        alloc = torch.cuda.memory_allocated() / 1e9
        on_gpu = all(t.device.type == "cuda" for t in sd.values())
        r.update(ssd_load_s=t_ssd, ssd_load_J=e_ssd, ssd_load_avg_W=e_ssd / t_ssd, ssd_bw_GBs=size / t_ssd,
                 gpu_allocated_GB=alloc, all_tensors_on_gpu=on_gpu)
        r["ssd_maybe_cached"] = size / t_ssd > 7.0
        if r["ssd_maybe_cached"]:
            print("  NOTE: faster than an NVMe SSD; the file probably came from the RAM file cache "
                  "(restart the computer before running to measure the SSD)")
        print(f"  CHECK: {alloc:.2f} GB allocated on the GPU for a {size:.2f} GB checkpoint; all tensors on cuda: "
              f"{on_gpu}  -> {'MODEL IS ON THE GPU' if on_gpu and alloc > 0.9 * size else 'PROBLEM: NOT ON GPU'}")

        # C. idle with the model resident
        r["P_resident_W"] = idle(meter, IDLE_SECONDS, "idle with the model resident")

        # D2. DRAM -> GPU: pinned host memory -> GPU for 25/50/75/100 % of the weights, 3 repetitions each
        cpu = {k: v.to("cpu").pin_memory() for k, v in sd.items()}
        del sd
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)
        keys = list(cpu)
        r["dram_points"] = []
        for frac in (0.25, 0.5, 0.75, 1.0):
            sub, acc = [], 0
            for k in keys:
                if acc >= frac * total:
                    break
                sub.append(k)
                acc += cpu[k].numel() * cpu[k].element_size()
            # warm-up transfer first (a laptop GPU needs ~0.1-0.2 s to leave its idle power state), then
            # repeat the transfer back-to-back for >= 2 s so that time and NVML energy are well resolved
            sd = {k: cpu[k].to("cuda", non_blocking=True) for k in sub}
            torch.cuda.synchronize()
            del sd
            n_rep, sec_needed = 0, 2.0
            with meter as m:
                t_start = time.perf_counter()
                while n_rep < 3 or time.perf_counter() - t_start < sec_needed:
                    sd = {k: cpu[k].to("cuda", non_blocking=True) for k in sub}
                    torch.cuda.synchronize()
                    del sd
                    n_rep += 1
            torch.cuda.empty_cache()
            sec, jl, aw = m.seconds / n_rep, m.joules / n_rep, m.avg_w
            r["dram_points"].append(dict(size_GB=acc / 1e9, s=sec, J=jl, avg_W=aw, window_s=m.seconds,
                                         repetitions=n_rep))
            print(f"  DRAM -> GPU {acc / 1e9:5.2f} GB: {sec:.3f} s per transfer ({n_rep} back-to-back), {jl:.1f} J, "
                  f"{acc / 1e9 / sec:.2f} GB/s")
        full = r["dram_points"][-1]
        r.update(dram_load_s=full["s"], dram_load_J=full["J"], dram_load_avg_W=full["avg_W"],
                 dram_bw_GBs=full["size_GB"] / full["s"])
        sd = None
        del sd, cpu
        gc.collect()
        torch.cuda.empty_cache()

        r["decode"] = []
        if INFERENCE_OK:
            try:
                # D3. full engine start (framework init + weights from the file cache), as a serverless cold start
                with meter as m:
                    tok = AutoTokenizer.from_pretrained(local[mid])
                    model = load_model(local[mid])
                    torch.cuda.synchronize()
                r.update(engine_start_s=m.seconds, engine_start_J=m.joules)
                print(f"  full model start (transformers): {m.seconds:.2f} s")
                dev = next(model.parameters()).device
                print(f"  CHECK: model parameters are on {dev}")
                r["model_device"] = str(dev)

                # E1. prefill
                ids = torch.randint(100, 20000, (1, PREFILL_TOKENS), device="cuda")
                with torch.no_grad():
                    model(ids)
                    torch.cuda.synchronize()
                    with meter as m:
                        for _ in range(5):
                            model(ids)
                        torch.cuda.synchronize()
                r["prefill_tps"] = 5 * PREFILL_TOKENS / m.seconds
                r["prefill_avg_W"] = m.avg_w
                print(f"  prefill: {r['prefill_tps']:.0f} tokens/s at {m.avg_w:.0f} W")

                # E2. decode at several batch sizes
                r["decode"] = []
                for b in BATCHES:
                    try:
                        x = torch.randint(100, 20000, (b, 64), device="cuda")
                        with torch.no_grad():
                            model.generate(x, max_new_tokens=8, min_new_tokens=8, do_sample=False,
                                           pad_token_id=tok.eos_token_id)
                            torch.cuda.synchronize()
                            with meter as m:
                                model.generate(x, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False,
                                               pad_token_id=tok.eos_token_id)
                                torch.cuda.synchronize()
                    except torch.cuda.OutOfMemoryError:
                        print(f"  batch {b}: out of memory, skipped")
                        torch.cuda.empty_cache()
                        continue
                    step = m.seconds / NEW_TOKENS
                    jpt = m.joules / (NEW_TOKENS * b)
                    r["decode"].append(dict(batch=b, step_s=step, avg_W=m.avg_w, J_per_token=jpt,
                                            J_per_token_above_bare=(m.avg_w - res["P_bare_W"]) * m.seconds / (NEW_TOKENS * b)))
                    print(f"  decode batch {b:>2}: {step * 1000:.1f} ms/step, {m.avg_w:.0f} W, {jpt:.3f} J/token")
                del model
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:  # noqa: BLE001
                print(f"  inference part skipped: {e!r}")
                r["inference_error"] = repr(e)
                gc.collect()
                torch.cuda.empty_cache()
        else:
            print("  inference part skipped (transformers cannot build the model; see the message at the start)")
        res["models"][mid] = r
        with open(os.path.join(OUT_DIR, "gpu_validation_partial.json"), "w") as f:
            json.dump(res, f, indent=2)

    calibrate(res)
    with open(os.path.join(OUT_DIR, "gpu_validation.json"), "w") as f:
        json.dump(res, f, indent=2)
    write_csv(res)
    print(f"\nSaved {OUT_DIR}/gpu_validation.json and {OUT_DIR}/gpu_validation.csv")


def _fit(points):
    """Least squares t = t_init + size / bw."""
    xs = [p["size_GB"] for p in points]
    ys = [p["s"] for p in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else my / max(mx, 1e-9)
    return max(my - slope * mx, 0.0), (1.0 / slope if slope > 0 else float("inf"))


def _p_ld(points, res):
    """Extra power during a load in the simulator's model: E_above_bare = (p_ctx + p_ld) * t."""
    e = sum(p["J"] - res["P_bare_W"] * p["s"] for p in points)
    t = sum(p["s"] for p in points)
    return e / t - res["context_overhead_W"]


def _loo(points, res, label):
    """Leave-one-out: fit the simulator's load model without one point, predict that point."""
    out = []
    for i, p in enumerate(points):
        rest = points[:i] + points[i + 1:]
        if len(rest) < 2:
            continue
        t0, bw = _fit(rest)
        pld = _p_ld(rest, res)
        t_pred = t0 + p["size_GB"] / bw
        e_pred = (res["context_overhead_W"] + pld) * t_pred
        e_meas = p["J"] - res["P_bare_W"] * p["s"]
        ok_e = p.get("window_s", p["s"]) >= MIN_ENERGY_WINDOW_S
        out.append(dict(tier=label, model=p["model"], size_GB=p["size_GB"], load_s_meas=p["s"], load_s_sim=t_pred,
                        load_s_err_pct=100 * (t_pred - p["s"]) / p["s"], load_J_meas_above_bare=e_meas,
                        load_J_sim=e_pred,
                        load_J_err_pct=100 * (e_pred - e_meas) / max(e_meas, 1e-9) if ok_e else None))
    return out


def calibrate(res):
    """Fit the simulator's parameters for this GPU and validate its load formulas out of sample."""
    ms = res["models"]
    pts = {"DRAM": [], "SSD": []}
    for k, m in ms.items():
        for tier, key in (("DRAM", "dram_points"), ("SSD", "ssd_points")):
            for p in m.get(key, []):
                pts[tier].append(dict(p, model=k.split("/")[-1]))
    card = dict(p_ctx_W=res["context_overhead_W"], bare_idle_W=res["P_bare_W"],
                resident_extra_W={k: v["P_resident_W"] - res["P_ctx_W"] for k, v in ms.items()})
    for tier in ("DRAM", "SSD"):
        if len(pts[tier]) >= 2:
            t0, bw = _fit(pts[tier])
            card[f"t_init_s_{tier}"], card[f"bw_{tier}_GBs"] = t0, bw
            card[f"p_load_extra_W_{tier}"] = _p_ld(pts[tier], res)
    big = max(ms.values(), key=lambda m: m["params_B"])
    dec = {d["batch"]: d for d in big.get("decode", [])}
    if "prefill_tps" in big:
        card["prefill_tps_7b"] = big["prefill_tps"] * big["params_B"] / 7.0
    if 1 in dec:
        card["step_s_7b"] = dec[1]["step_s"] * 7.0 / big["params_B"]
        if len(dec) > 1:
            bmax = max(dec)
            card["step_slope"] = (dec[bmax]["step_s"] / dec[1]["step_s"] - 1) / (bmax - 1)
            card["p_dyn_decode_W"] = dec[bmax]["avg_W"] - res["P_ctx_W"]
    comp = _loo(pts["DRAM"], res, "DRAM") + _loo(pts["SSD"], res, "SSD")
    res["calibration_card"], res["formula_check"] = card, comp
    print("\n================ CALIBRATION CARD (simulator parameters for this GPU) ================")
    for k, v in card.items():
        print(f"  {k:<22} {v}")
    print("\n====== SIMULATOR LOAD MODEL vs MEASUREMENT (leave-one-out: each point predicted from the others) ======")
    for c in comp:
        print(f"  {c['tier']:<4} {c['model']:<22} {c['size_GB']:5.2f} GB  time {c['load_s_meas']:.2f}s vs "
              f"{c['load_s_sim']:.2f}s ({c['load_s_err_pct']:+.0f}%)  energy " +
              (f"{c['load_J_meas_above_bare']:.1f}J vs {c['load_J_sim']:.1f}J ({c['load_J_err_pct']:+.0f}%)"
               if c["load_J_err_pct"] is not None else "(window too short for energy)"))
    if comp:
        for tier in ("DRAM", "SSD"):
            e = [abs(c["load_s_err_pct"]) for c in comp if c["tier"] == tier]
            f = [abs(c["load_J_err_pct"]) for c in comp if c["tier"] == tier and c["load_J_err_pct"] is not None]
            if e:
                fe = f"{sum(f) / len(f):.1f}%" if f else "n/a (windows too short for the NVML energy counter)"
                print(f"  {tier}: mean |error| time {sum(e) / len(e):.1f}%, energy {fe}")
    print("\n  Context overhead independent of model size? resident-minus-context power per model:",
          {k.split('/')[-1]: round(v, 1) for k, v in card["resident_extra_W"].items()}, "W (should be ~0)")


def write_csv(res):
    rows = [("gpu", res["gpu"]), ("P_bare_W", res["P_bare_W"]), ("P_ctx_W", res["P_ctx_W"]),
            ("context_overhead_W", res["context_overhead_W"])]
    for k, v in res["calibration_card"].items():
        rows.append((f"card.{k}", v))
    for mid, m in res["models"].items():
        for k, v in m.items():
            if k not in ("decode", "ssd_points", "dram_points"):
                rows.append((f"{mid}.{k}", v))
        for key in ("ssd_points", "dram_points"):
            for j, pnt in enumerate(m.get(key, [])):
                for kk, vv in pnt.items():
                    rows.append((f"{mid}.{key}{j}.{kk}", vv))
        for d in m.get("decode", []):
            for k, v in d.items():
                rows.append((f"{mid}.decode_b{d['batch']}.{k}", v))
    with open(os.path.join(OUT_DIR, "gpu_validation.csv"), "w", encoding="utf-8") as f:
        f.write("quantity,value\n")
        for k, v in rows:
            f.write(f'{k},"{v}"\n')


if __name__ == "__main__":
    main()
