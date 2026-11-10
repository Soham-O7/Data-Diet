"""Item 4: confirm the real CodeCarbon backend works and compare it with the CPU-time estimate.

  pip install codecarbon
  python scripts/check_codecarbon.py --seconds 8 --grid-kg-per-kwh 0.7 --price-per-kwh 8

Runs the SAME busy workload through EnergyMeter twice (estimate, then codecarbon). Use an idle machine.
CodeCarbon samples about once a second, so stages shorter than a few seconds are noise.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataprune.energy import EnergyMeter  # noqa: E402


def burn(seconds: float):
    t0 = time.perf_counter()
    x = 0
    while time.perf_counter() - t0 < seconds:
        x += sum(i * i for i in range(20000))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--watts-per-core", type=float, default=15.0)
    ap.add_argument("--grid-kg-per-kwh", type=float, default=0.7)
    ap.add_argument("--price-per-kwh", type=float, default=8.0)
    ap.add_argument("--country", default="IND", help="ISO code for codecarbon's offline tracker")
    a = ap.parse_args()
    out = {}
    for backend in ("estimate", "codecarbon"):
        m = EnergyMeter(a.watts_per_core, backend, a.grid_kg_per_kwh, a.price_per_kwh, a.country)
        with m.stage("burn"):
            burn(a.seconds)
        s = m.summary()
        out[backend] = s
        print(f"[{backend:<10}] backend_used={s['backend_used']:<10} kWh={s['total_kwh_est']:.3e} "
              f"CO2={s['total_co2_kg_est']:.3e} kg  cost={s['total_cost_est']:.3e}  note={s['backend_note'] or '-'}")
    if out["codecarbon"]["backend_used"] != "codecarbon":
        print("\nFAIL: codecarbon was not used (see note). Install it with `pip install codecarbon`.")
        return 1
    e, c = out["estimate"]["total_kwh_est"], out["codecarbon"]["total_kwh_est"]
    print(f"\nratio codecarbon/estimate = {c / e:.2f}  (1.0 = agree; the estimate assumes {a.watts_per_core} W per busy core)")
    print("If the ratio is far from 1, set --watts-per-core to your CPU's real draw, or trust CodeCarbon.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
