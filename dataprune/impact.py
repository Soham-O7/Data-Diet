"""Cumulative impact tracker (kWh, CO2, cost across runs) and 'bring your own model' energy measurement.

Both use the same estimate as the rest of the package (child CPU-seconds x watts/core) unless codecarbon is used
inside your own script and you pass its readings to `ImpactTracker.log`. Estimates are not meter readings.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence


class ImpactTracker:
    """Append-only JSON log of runs with running totals. `log(...)` takes kWh numbers you already have."""

    def __init__(self, path: str = "impact_log.json"):
        self.path = path

    def _load(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return json.load(f)

    def log(self, label: str, overhead_kwh: float, saved_kwh_per_run: float, runs: int = 1,
            grid_kg_per_kwh: float = 0.7, price_per_kwh: float = 8.0, source: str = "estimate") -> dict:
        """Net saving = saved_kwh_per_run * runs - overhead_kwh (negative = pruning has not paid off yet)."""
        net = saved_kwh_per_run * runs - overhead_kwh
        rec = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": label, "runs": runs,
               "overhead_kwh": overhead_kwh, "saved_kwh_per_run": saved_kwh_per_run, "net_kwh": net,
               "net_co2_kg": net * grid_kg_per_kwh, "net_cost": net * price_per_kwh, "source": source}
        data = self._load()
        data.append(rec)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)
        return rec

    def log_experiment(self, label: str, out: dict, runs: int = 1, cfg=None) -> dict:
        """Log from a `run_experiment` output (uses its verdict['savings_card'])."""
        card = out["verdict"]["savings_card"]
        g = cfg.grid_kg_co2_per_kwh if cfg else 0.7
        p = cfg.price_per_kwh if cfg else 8.0
        return self.log(label, card["pruning_overhead_kwh"], card["per_run_saving_kwh"], runs, g, p)

    def totals(self) -> dict:
        d = self._load()
        return {"n_logged": len(d), "net_kwh": sum(r["net_kwh"] for r in d),
                "net_co2_kg": sum(r["net_co2_kg"] for r in d), "net_cost": sum(r["net_cost"] for r in d),
                "overhead_kwh": sum(r["overhead_kwh"] for r in d),
                "note": "sums of ESTIMATES; negative = overhead not yet repaid"}


def _windows_cpu_seconds(proc) -> float:
    """User+kernel CPU seconds of the directly launched process (GetProcessTimes). Fallback only."""
    import ctypes
    from ctypes import wintypes

    ft = [wintypes.FILETIME() for _ in range(4)]       # creation, exit, kernel, user
    ok = ctypes.windll.kernel32.GetProcessTimes(wintypes.HANDLE(int(proc._handle)), *[ctypes.byref(t) for t in ft])
    if not ok:
        raise OSError("GetProcessTimes failed")
    to_s = lambda t: ((t.dwHighDateTime << 32) | t.dwLowDateTime) / 1e7      # 100 ns ticks
    return to_s(ft[2]) + to_s(ft[3])


def _windows_run_in_job(cmd, cwd, timeout):
    """Run cmd inside a Job Object so CPU time of ALL descendants counts (a venv's python.exe is only a launcher
    that starts the real interpreter as a child). Returns (returncode, cpu_seconds)."""
    import ctypes
    from ctypes import wintypes

    k32, ntdll = ctypes.windll.kernel32, ctypes.windll.ntdll
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
                                              wintypes.LPVOID]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]

    class Accounting(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise OSError("CreateJobObject failed")
    try:
        proc = subprocess.Popen(list(cmd), cwd=cwd, creationflags=0x00000004)         # CREATE_SUSPENDED
        handle = wintypes.HANDLE(int(proc._handle))
        try:
            assigned = k32.AssignProcessToJobObject(job, handle)
        finally:
            ntdll.NtResumeProcess(handle)                                              # always let it run
        if not assigned:
            proc.wait(timeout=timeout)
            raise OSError("AssignProcessToJobObject failed")
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise
        info = Accounting()
        if not k32.QueryInformationJobObject(job, 1, ctypes.byref(info), ctypes.sizeof(info), None):   # 1 = BasicAccounting
            raise OSError("QueryInformationJobObject failed")
        return proc.returncode, (info.TotalUserTime + info.TotalKernelTime) / 1e7
    finally:
        k32.CloseHandle(job)


def measure_command(cmd: Sequence[str], watts_per_core: float = 15.0, cwd: Optional[str] = None,
                    timeout: Optional[float] = None) -> Dict[str, float]:
    """Run a training command and return wall time, CPU-seconds of the child and estimated kWh.

    Linux/macOS: CPU time of the whole child process tree (RUSAGE_CHILDREN delta).
    Windows: CPU time of the whole process tree via a Job Object (falls back to the launched process only if job
    objects are unavailable). GPU draw is NOT included either way. Run on an otherwise idle machine.
    """
    t0 = time.perf_counter()
    if os.name == "nt":
        try:
            rc, cpu = _windows_run_in_job(cmd, cwd, timeout)
        except OSError:                      # job objects unavailable: single-process fallback (may undercount)
            proc = subprocess.Popen(list(cmd), cwd=cwd)
            proc.wait(timeout=timeout)
            rc, cpu = proc.returncode, _windows_cpu_seconds(proc)
        wall = time.perf_counter() - t0

        class proc:                          # noqa: N801 - tiny stand-in so the return line below is shared
            returncode = rc
    else:
        import resource

        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        proc = subprocess.run(list(cmd), cwd=cwd, timeout=timeout)
        wall = time.perf_counter() - t0
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return {"returncode": proc.returncode, "wall_s": wall, "cpu_s": cpu, "kwh_est": cpu * watts_per_core / 3.6e6}


def compare_training(full_cmd: Sequence[str], pruned_cmd: Sequence[str], overhead_kwh: float = 0.0,
                     watts_per_core: float = 15.0, grid_kg_per_kwh: float = 0.7, price_per_kwh: float = 8.0) -> dict:
    """Run the user's own training command on the full data and on the pruned data; report measured saving.

    Your script decides which data it reads (e.g. a path argument), so the two commands differ only in that.
    """
    from .energy import savings_card

    full = measure_command(full_cmd, watts_per_core)
    pruned = measure_command(pruned_cmd, watts_per_core)
    for name, r in (("full", full), ("pruned", pruned)):
        if r["returncode"] != 0:
            raise RuntimeError(f"{name} training command exited with code {r['returncode']}")
    card = savings_card(overhead_kwh, full["kwh_est"], pruned["kwh_est"], grid_kg_per_kwh, price_per_kwh)
    return {"full": full, "pruned": pruned, "savings_card": card,
            "note": "CPU-time based estimate of YOUR model's training cost (no GPU); idle machine assumed."}


def python_cmd(script: str, *args: str) -> List[str]:
    return [sys.executable, script, *args]