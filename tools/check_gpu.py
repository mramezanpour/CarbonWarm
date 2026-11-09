#!/usr/bin/env python3
"""Check that PyTorch uses the NVIDIA GPU.  Usage: python tools/check_gpu.py"""
import subprocess
import sys
import time

FAILED = []


def ok(msg):
    print(f"[OK]   {msg}")


def bad(msg):
    print(f"[FAIL] {msg}")
    FAILED.append(msg)


def main():
    # 1. PyTorch with CUDA support
    try:
        import torch
    except ImportError:
        bad("PyTorch is not installed")
        return finish()
    ok(f"PyTorch {torch.__version__} is installed")
    if torch.version.cuda is None:
        bad("this PyTorch build has NO CUDA support (it is the CPU-only build); install the CUDA build")
        return finish()
    ok(f"PyTorch was built with CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        bad("torch.cuda.is_available() is False: no NVIDIA GPU / driver visible to PyTorch")
        return finish()
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    ok(f"GPU found: {name}, {vram:.1f} GB memory")

    # 2. memory really allocated on the GPU
    before = torch.cuda.memory_allocated(0)
    x = torch.randn(8192, 8192, device=dev)          # 256 MB
    y = torch.randn(8192, 8192, device=dev)
    torch.cuda.synchronize()
    used = (torch.cuda.memory_allocated(0) - before) / 2**20
    if x.device.type == "cuda" and used > 400:
        ok(f"tensors live on the GPU ({x.device}), {used:.0f} MB allocated in GPU memory")
    else:
        bad(f"tensors are not on the GPU (device={x.device}, allocated {used:.0f} MB)")

    # 3. the GPU is much faster than the CPU on a matrix product
    torch.matmul(x, y)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        torch.matmul(x, y)
    torch.cuda.synchronize()
    t_gpu = (time.perf_counter() - t) / 10
    xc, yc = x[:2048, :2048].cpu(), y[:2048, :2048].cpu()
    t = time.perf_counter()
    torch.matmul(xc, yc)
    t_cpu = (time.perf_counter() - t) * (8192 / 2048) ** 3      # scaled to the same size
    speed = t_cpu / t_gpu
    msg = f"8192x8192 matrix product: GPU {t_gpu * 1000:.1f} ms vs CPU ~{t_cpu * 1000:.0f} ms ({speed:.0f}x faster)"
    (ok if speed > 5 else bad)(msg)

    # 4. NVML (power readings) works and shows this process on the GPU
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        ok(f"NVML power reading works: {p:.1f} W now")
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            import os
            mine = [q for q in procs if q.pid == os.getpid()]
            if mine:
                ok(f"NVML lists this Python process on the GPU (pid {os.getpid()})")
            else:
                print("[INFO] NVML does not list this process (normal on some Windows drivers)")
        except pynvml.NVMLError:
            print("[INFO] process list not available from NVML (normal on Windows)")
    except ImportError:
        bad("package nvidia-ml-py is missing: pip install nvidia-ml-py (needed for power measurements)")
    except Exception as e:  # noqa: BLE001
        bad(f"NVML does not work: {e!r}")

    # 5. nvidia-smi as an independent check
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,power.draw", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        if out:
            ok(f"nvidia-smi: {out}  (memory.used includes the tensors of this script)")
    except Exception:  # noqa: BLE001
        print("[INFO] nvidia-smi not found on PATH (not required)")
    del x, y
    return finish()


def finish():
    print()
    if FAILED:
        print("GPU CHECK FAILED:", "; ".join(FAILED))
    else:
        print("GPU CHECK PASSED: PyTorch computes on the GPU and power can be measured.")
    return not FAILED


if __name__ == "__main__":
    passed = main()
    if "IPython" not in sys.modules:
        sys.exit(0 if passed else 1)
