"""Compute / energy accounting.

IMPORTANT: this is an ESTIMATE. Energy = CPU-seconds x assumed watts per core. It does not read a power meter
and it does not see GPU draw. For real numbers on your hardware, set `cpu_watts_per_core` from your CPU's TDP
or wrap the run with a real meter (e.g. codecarbon / NVML) and plug the measured Joules into `break_even_runs`.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Any


@dataclass
class StageCost:
    wall_s: float = 0.0
    cpu_s: float = 0.0


class EnergyMeter:
    """Per-stage compute accounting.

    backend="estimate"    : CPU-seconds x watts_per_core (default; always works).
    backend="codecarbon"  : each stage is wrapped in a codecarbon OfflineEmissionsTracker and its measured kWh is used.
                            Falls back to the estimate (and says so in `backend_used` / `backend_note`) if codecarbon is
                            not installed or fails. codecarbon samples power about once per `measure_power_secs`, so very
                            short stages (< ~1 s) can read near zero - treat those as noise, not savings.
    CO2 = kWh x grid_kg_per_kwh and cost = kWh x price_per_kwh. Both defaults are PLACEHOLDERS: set your local
    grid carbon intensity and tariff.
    """

    def __init__(self, watts_per_core: float = 15.0, backend: str = "estimate", grid_kg_per_kwh: float = 0.7,
                 price_per_kwh: float = 8.0, country_iso_code: str = "IND", measure_power_secs: int = 1):
        self.watts_per_core = float(watts_per_core)
        self.grid_kg_per_kwh = float(grid_kg_per_kwh)
        self.price_per_kwh = float(price_per_kwh)
        self.country_iso_code = country_iso_code
        self.measure_power_secs = measure_power_secs
        self.stages: Dict[str, StageCost] = {}
        self._measured_kwh: Dict[str, float] = {}
        self._cc = None
        self.backend_used = "estimate"
        self.backend_note = ""
        if backend == "codecarbon":
            try:
                import codecarbon  # type: ignore

                self._cc = codecarbon
                self.backend_used = "codecarbon"
            except Exception as e:  # not installed
                self.backend_note = f"codecarbon unavailable ({type(e).__name__}); using CPU-time estimate"
        elif backend != "estimate":
            raise ValueError("backend must be 'estimate' or 'codecarbon'")

    @contextmanager
    def stage(self, name: str):
        tracker = None
        if self._cc is not None:
            try:
                tracker = self._cc.OfflineEmissionsTracker(
                    country_iso_code=self.country_iso_code, save_to_file=False, log_level="error",
                    measure_power_secs=self.measure_power_secs)
                tracker.start()
            except Exception as e:
                tracker = None
                self.backend_note = f"codecarbon tracker failed ({type(e).__name__}); stage '{name}' estimated"
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            cost = self.stages.setdefault(name, StageCost())
            cost.wall_s += time.perf_counter() - w0
            cost.cpu_s += time.process_time() - c0
            if tracker is not None:
                try:
                    tracker.stop()
                    data = getattr(tracker, "final_emissions_data", None)
                    kwh = getattr(data, "energy_consumed", None)
                    if kwh is not None:
                        self._measured_kwh[name] = self._measured_kwh.get(name, 0.0) + float(kwh)
                except Exception as e:
                    self.backend_note = f"codecarbon stop failed ({type(e).__name__}); stage '{name}' estimated"

    # --- derived numbers -------------------------------------------------
    def _stage_kwh(self, name: str) -> float:
        if name in self._measured_kwh:
            return self._measured_kwh[name]
        return self.stages[name].cpu_s * self.watts_per_core / 3.6e6

    def kwh(self, name: str | None = None) -> float:
        if name is not None:
            return self._stage_kwh(name)
        return sum(self._stage_kwh(k) for k in self.stages)

    def joules(self, name: str | None = None) -> float:
        return self.kwh(name) * 3.6e6

    def co2_kg(self, name: str | None = None) -> float:
        return self.kwh(name) * self.grid_kg_per_kwh

    def cost(self, name: str | None = None) -> float:
        return self.kwh(name) * self.price_per_kwh

    @property
    def total_wall_s(self) -> float:
        return sum(s.wall_s for s in self.stages.values())

    @property
    def total_cpu_s(self) -> float:
        return sum(s.cpu_s for s in self.stages.values())

    def summary(self) -> dict:
        return {
            "estimate_note": ("measured by codecarbon" if self.backend_used == "codecarbon"
                              else "CPU-seconds x assumed watts/core; not a power-meter reading"),
            "backend_used": self.backend_used,
            "backend_note": self.backend_note,
            "watts_per_core": self.watts_per_core,
            "grid_kg_co2_per_kwh": self.grid_kg_per_kwh,
            "price_per_kwh": self.price_per_kwh,
            "total_wall_s": self.total_wall_s,
            "total_cpu_s": self.total_cpu_s,
            "total_kwh_est": self.kwh(),
            "total_co2_kg_est": self.co2_kg(),
            "total_cost_est": self.cost(),
            "stages": {
                k: {"wall_s": v.wall_s, "cpu_s": v.cpu_s, "kwh_est": self._stage_kwh(k)}
                for k, v in self.stages.items()
            },
        }


def savings_card(overhead_kwh: float, full_train_kwh: float, pruned_train_kwh: float,
                 grid_kg_per_kwh: float = 0.7, price_per_kwh: float = 8.0, runs: int = 1) -> dict:
    """Per-run and over-`runs` savings of pruned vs full training, net of the one-off pruning overhead."""
    per_run = full_train_kwh - pruned_train_kwh
    be = break_even_runs(overhead_kwh * 3.6e6, full_train_kwh * 3.6e6, pruned_train_kwh * 3.6e6)
    net_kwh = per_run * runs - overhead_kwh
    return {
        "per_run_saving_kwh": per_run,
        "per_run_saving_co2_kg": per_run * grid_kg_per_kwh,
        "per_run_saving_cost": per_run * price_per_kwh,
        "pruning_overhead_kwh": overhead_kwh,
        "pruning_overhead_co2_kg": overhead_kwh * grid_kg_per_kwh,
        "runs": runs,
        "net_saving_kwh": net_kwh,
        "net_saving_co2_kg": net_kwh * grid_kg_per_kwh,
        "net_saving_cost": net_kwh * price_per_kwh,
        "break_even_runs": None if math.isinf(be) else be,
    }


def measure(fn: Callable[..., Any], *args, watts_per_core: float = 15.0, **kwargs) -> Tuple[Any, StageCost, float]:
    """Run fn, return (result, StageCost, estimated_joules)."""
    w0, c0 = time.perf_counter(), time.process_time()
    out = fn(*args, **kwargs)
    cost = StageCost(time.perf_counter() - w0, time.process_time() - c0)
    return out, cost, cost.cpu_s * watts_per_core


def break_even_runs(overhead_j: float, full_train_j: float, pruned_train_j: float) -> float:
    """Number of training runs after which the pruning overhead has paid for itself.

    overhead_j: energy spent by the pruning pipeline (INCLUDING embeddings / proxy training).
    full_train_j / pruned_train_j: energy of ONE training run on full vs pruned data.
    Returns math.inf if pruning does not save energy per run.
    """
    saving = full_train_j - pruned_train_j
    if saving <= 0:
        return math.inf
    return overhead_j / saving
