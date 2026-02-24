import json
import os
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pyomo.environ import (
    ConcreteModel,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Var,
    maximize,
)
from pyomo.opt import SolverFactory, TerminationCondition

from energbench.utils import generate_timestamp

from .base_tool import BaseTool, tool_method
from .constants import (
    BATTERY_CSV_MAX_FILE_SIZE_MB,
    BATTERY_CSV_MAX_ROWS,
    BATTERY_INITIAL_SOC_FRACTION,
    BATTERY_ROUNDING_PROFILE,
    BATTERY_ROUNDING_SUMMARY,
)


class BatteryOptimizationTool(BaseTool):
    """Tool for battery storage optimization."""

    def __init__(self) -> None:
        super().__init__(
            name="battery_optimization",
            description="Optimize battery storage operations for maximum revenue",
        )

    @staticmethod
    def _load_csv(csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
        size_bytes = os.path.getsize(csv_path)
        max_bytes = BATTERY_CSV_MAX_FILE_SIZE_MB * 1024 * 1024
        if size_bytes > max_bytes:
            raise ValueError(
                f"CSV file is {size_bytes / 1024 / 1024:.1f} MB, "
                f"exceeding the {BATTERY_CSV_MAX_FILE_SIZE_MB} MB limit."
            )
        df = pd.read_csv(csv_path)
        if len(df) > BATTERY_CSV_MAX_ROWS:
            raise ValueError(
                f"CSV has {len(df):,} rows, exceeding the {BATTERY_CSV_MAX_ROWS:,} row limit."
            )
        return df

    @tool_method()
    def battery_revenue_optimization(
        self,
        run_description: str,
        csv_path: str,
        energy_price_column: str,
        battery_size_mw: float,
        battery_duration: float,
        battery_degradation_cost: float = 24.0,
        round_trip_efficiency: float = 0.81,
        minimum_state_of_charge: float = 0.1,
        maximum_state_of_charge: float = 0.8,
        timestep_in_hours: float = 1.0,
        days: float = 365.0,
    ) -> str:
        """Produces optimal potential net revenues for battery storage projects using energy arbitrage only
        (perfect price foresight, no price-uncertainty penalty). Reads a CSV of energy prices and returns
        summary metrics, time-series rows, and a saved CSV path.

        Args:
            run_description: One word description of the model run, used for the filename of the output csv file
            csv_path: Path to the CSV file containing the energy prices.
            energy_price_column: Column name of the energy prices.
            battery_size_mw: Battery size in MW. Battery size is the maximum power output of the battery.
            battery_duration: Battery duration in hours. Battery duration is the maximum amount of time the battery can store energy.
            battery_degradation_cost: cost in $/MWh reflecting additional costs to preserve battery life based on number
                                    of cycles and capital cost. Default is $24/MWh based on $110,000/MW and $205,000/MWh
                                    default capex and 6000 cycles
            round_trip_efficiency: float reflecting battery charging and discharging efficiency. Should always be between 0 and 1
            minimum_state_of_charge: float reflecting battery minimum state of charge. Should always be between 0 and 1
            maximum_state_of_charge: float reflecting battery maximum state of charge. Should always be between 0 and 1
            timestep_in_hours: float representing minimum time step
            days: float representing number of days in the input price data

        Returns:
            JSON string with the optimization results. The value fields are as follows.

            "run_description": description of the run,
            "net_revenue": float reflecting net battery revenues after removing charging and degradation costs in $
            "total_revenue": float reflecting total battery discharge revenues in $
            "charging_cost": float reflecting annual charging costs in $
            "degradation_penalty": float reflecting additional O&M for degradation management in $
            "rows": contains time series results including operating profile, state_of_charge, net_revenue, total_revenue,
                    charging_cost an degradation penalty
            "saved_csv": saved file path for the output csv

        """
        if battery_size_mw <= 0:
            return json.dumps({"error": "battery_size_mw must be positive."})
        if battery_duration <= 0:
            return json.dumps({"error": "battery_duration must be positive."})
        if timestep_in_hours <= 0:
            return json.dumps({"error": "timestep_in_hours must be positive."})
        if not (0 < round_trip_efficiency <= 1):
            return json.dumps({"error": "round_trip_efficiency must be in (0, 1]."})
        if not (0 <= minimum_state_of_charge < 1):
            return json.dumps({"error": "minimum_state_of_charge must be in [0, 1)."})
        if not (0 < maximum_state_of_charge <= 1):
            return json.dumps({"error": "maximum_state_of_charge must be in (0, 1]."})
        if minimum_state_of_charge >= maximum_state_of_charge:
            return json.dumps({"error": "minimum_state_of_charge must be less than maximum_state_of_charge."})
        try:
            df = self._load_csv(csv_path)
            if energy_price_column not in df.columns:
                return json.dumps({"error": f"Column '{energy_price_column}' not found in CSV."})

            energy_price = df[energy_price_column].values
            horizon = int(days * 24 / timestep_in_hours)

            if horizon > len(energy_price):
                return json.dumps({
                    "error": (
                        f"Requested horizon ({horizon} intervals = {days} days) "
                        f"exceeds available price data ({len(energy_price)} intervals). "
                        f"Reduce 'days' or provide a longer CSV."
                    ),
                })

            battery_size_mwh = battery_size_mw * battery_duration
            charge_eff = np.sqrt(round_trip_efficiency)
            discharge_eff = np.sqrt(round_trip_efficiency)

            model = ConcreteModel()
            model.storage_soc = Var(range(horizon), domain=NonNegativeReals)
            model.charge_power = Var(range(horizon), domain=NonNegativeReals)
            model.discharge_power = Var(range(horizon), domain=NonNegativeReals)
            model.abs_power = Var(range(horizon), domain=NonNegativeReals)
            model.constraints = ConstraintList()

            for t in range(horizon):
                model.constraints.add(model.storage_soc[t] >= minimum_state_of_charge * battery_size_mwh)
                model.constraints.add(model.storage_soc[t] <= maximum_state_of_charge * battery_size_mwh)
                model.constraints.add(model.charge_power[t] <= battery_size_mw)
                model.constraints.add(model.discharge_power[t] <= battery_size_mw)
                if t > 0:
                    model.constraints.add(
                        model.storage_soc[t]
                        == model.storage_soc[t - 1]
                        + timestep_in_hours
                        * ((charge_eff * model.charge_power[t]) - ((1 / discharge_eff) * model.discharge_power[t]))
                    )
                model.constraints.add(model.charge_power[t] - model.discharge_power[t] <= model.abs_power[t])
                model.constraints.add(model.discharge_power[t] - model.charge_power[t] <= model.abs_power[t])

            model.constraints.add(model.storage_soc[0] == BATTERY_INITIAL_SOC_FRACTION * battery_size_mwh)
            model.constraints.add(model.storage_soc[0] == model.storage_soc[horizon - 1])

            def total_cost(m: ConcreteModel) -> Any:
                energy_rev = sum(
                    (timestep_in_hours * energy_price[t] * (m.discharge_power[t] - m.charge_power[t]))
                    for t in range(horizon)
                )
                degradation_penalty = sum(
                    (timestep_in_hours * battery_degradation_cost * m.abs_power[t]) for t in range(horizon)
                )
                return energy_rev - degradation_penalty

            model.cost = Objective(rule=total_cost, sense=maximize)

            solver = SolverFactory("ipopt")
            if solver is None or not solver.available():
                return json.dumps({"error": "IPOPT solver not available. Install it to run optimization."})
            results = solver.solve(model, report_timing=True)
            termination = results.solver.termination_condition
            if termination not in {
                TerminationCondition.optimal,
                TerminationCondition.feasible,
                TerminationCondition.locallyOptimal,
            }:
                return json.dumps(
                    {
                        "error": "Optimization did not converge to a feasible solution.",
                        "termination_condition": str(termination),
                        "solver_status": str(results.solver.status),
                    }
                )

            if any(model.discharge_power[t].value is None for t in range(horizon)):
                return json.dumps(
                    {
                        "error": "Optimization produced incomplete results.",
                        "termination_condition": str(termination),
                        "solver_status": str(results.solver.status),
                    }
                )

            total_revenue_series = [
                timestep_in_hours * energy_price[t] * model.discharge_power[t].value for t in range(horizon)
            ]
            total_revenue = float(np.sum(total_revenue_series))
            degradation_penalty_series = [
                -1 * timestep_in_hours * battery_degradation_cost * model.abs_power[t].value for t in range(horizon)
            ]
            degradation_penalty = float(np.sum(degradation_penalty_series))

            operation_profile = [
                model.discharge_power[t].value - model.charge_power[t].value for t in range(horizon)
            ]
            charging_only = np.minimum(0, np.array(operation_profile))
            charging_cost_series = timestep_in_hours * (charging_only * energy_price[: len(charging_only)])
            charging_cost = float(-1 * np.sum(charging_only * energy_price[: len(charging_only)]) * timestep_in_hours)

            net_revenue_series = (
                np.array(total_revenue_series) + charging_cost_series + np.array(degradation_penalty_series) #adding because degradation and charging cost series are negative
            )
            net_revenue = float(np.sum(net_revenue_series))

            state_of_charge = [model.storage_soc[t].value / battery_size_mwh for t in range(horizon)]

            output_rows = []
            for idx in range(horizon):
                output_rows.append(
                    {
                        "operation_profile": round(operation_profile[idx], BATTERY_ROUNDING_PROFILE),
                        "state_of_charge": round(state_of_charge[idx], BATTERY_ROUNDING_PROFILE),
                        "net_revenue": round(float(net_revenue_series[idx]), BATTERY_ROUNDING_PROFILE),
                        "total_revenue": round(float(total_revenue_series[idx]), BATTERY_ROUNDING_PROFILE),
                        "charging_cost": round(float(charging_cost_series[idx]), BATTERY_ROUNDING_PROFILE),
                        "degradation_penalty": round(float(degradation_penalty_series[idx]), BATTERY_ROUNDING_PROFILE),
                    }
                )

            timestamp = generate_timestamp()
            save_csv_path = f"battery_revenue_optimization_{timestamp}.csv"
            pd.DataFrame(output_rows).to_csv(save_csv_path, index=False)

            return json.dumps(
                {
                    "run_description": run_description,
                    "net_revenue": round(net_revenue, BATTERY_ROUNDING_SUMMARY),
                    "total_revenue": round(total_revenue, BATTERY_ROUNDING_SUMMARY),
                    "charging_cost": round(charging_cost, BATTERY_ROUNDING_SUMMARY),
                    "degradation_penalty": round(degradation_penalty, BATTERY_ROUNDING_SUMMARY),
                    "row_count": len(output_rows),
                    "saved_csv": save_csv_path,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Battery revenue optimization failed: {e}")
            return json.dumps({"error": str(e)})
