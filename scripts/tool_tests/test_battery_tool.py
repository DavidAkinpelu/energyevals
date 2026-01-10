#!/usr/bin/env python
"""Exercise BatteryOptimizationTool."""
from __future__ import annotations

import csv
import tempfile

from energbench.tools.battery_tool import BatteryOptimizationTool


def _write_price_csv(path: str) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["price"])
        for hour in range(24):
            writer.writerow([20 + hour])


def _print_result(label: str, result: str) -> None:
    print(f"\n=== {label} ===")
    print(result)


def main() -> None:
    tool = BatteryOptimizationTool()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        csv_path = handle.name

    _write_price_csv(csv_path)

    result = tool.battery_revenue_optimization(
        run_description="test",
        csv_path=csv_path,
        energy_price_column="price",
        battery_size_mw=1.0,
        battery_duration=2.0,
        battery_degradation_cost=5.0,
        round_trip_efficiency=0.9,
        minimum_state_of_charge=0.1,
        maximum_state_of_charge=0.9,
        timestep_in_hours=1.0,
        days=1.0,
    )
    _print_result("battery_revenue_optimization", result)


if __name__ == "__main__":
    main()
