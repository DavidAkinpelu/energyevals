"""Battery optimization tool for energy storage revenue stacking."""

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger

from energbench.agent.providers import ToolDefinition

from .base_tool import BaseTool


class BatteryOptimizationTool(BaseTool):
    """Tool for battery storage optimization."""

    def __init__(self):
        super().__init__(
            name="battery_optimization",
            description="Optimize battery storage operations for maximum revenue",
        )

        self.register_method("battery_revenue_optimization", self.battery_revenue_optimization)

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="battery_revenue_optimization",
                description=(
                    "Optimize battery revenue from energy arbitrage using Pyomo. "
                    "Reads a CSV of energy prices and returns revenue metrics and dispatch."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "run_description": {"type": "string",
                            "description": "One word description of the model run, used for the filename of the output csv file"
                        },
                        "csv_path": {"type": "string",
                            "description": "Path to the CSV file containing the energy prices."
                        },
                        "energy_price_column": {"type": "string",
                            "description": "Column name of the energy prices."
                        },
                        "battery_size_mw": {"type": "number",
                            "description": "Battery size in MW. Battery size is the maximum power output of the battery."
                        },
                        "battery_duration": {"type": "number",
                            "description": "Battery duration in hours. Battery duration is the maximum amount of time the battery can store energy."
                        },
                        "battery_degradation_cost": {"type": "number", "default": 24.0,
                            "description": "Battery degradation cost in $/MWh. Battery degradation cost is the cost of degradation of the battery."
                        },
                        "round_trip_efficiency": {"type": "number", "default": 0.81,
                            "description": "Round trip efficiency. Round trip efficiency is the efficiency of the battery when charging and discharging."
                        },
                        "minimum_state_of_charge": {"type": "number", "default": 0.1,
                            "description": "Minimum state of charge. Minimum state of charge is the minimum state of charge of the battery."
                        },
                        "maximum_state_of_charge": {"type": "number", "default": 0.8,
                            "description": "Maximum state of charge. Maximum state of charge is the maximum state of charge of the battery."
                        },
                        "timestep_in_hours": {"type": "number", "default": 1.0,
                            "description": "Timestep in hours. Timestep is the time step of the optimization."
                        },
                        "days": {"type": "number", "default": 365.0,
                            "description": "Number of days. Number of days is the number of days of the optimization."
                        },
                    },
                    "required": [
                        "run_description",
                        "csv_path",
                        "energy_price_column",
                        "battery_size_mw",
                        "battery_duration",
                    ],
                },

            ),

        ]

    @staticmethod
    def _load_csv(csv_path: str):
        import pandas as pd

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
        return pd.read_csv(csv_path)

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
        """Optimize battery revenue from energy arbitrage using Pyomo.
        Args:
            run_description: One word description of the model run, used for the filename of the output csv file
            csv_path: Path to the CSV file containing the energy prices.
            energy_price_column: Column name of the energy prices.
            battery_size_mw: Battery size in MW. Battery size is the maximum power output of the battery.
            battery_duration: Battery duration in hours. Battery duration is the maximum amount of time the battery can store energy.
            battery_degradation_cost: Battery degradation cost in $/MWh. Battery degradation cost is the cost of degradation of the battery.
            round_trip_efficiency: Round trip efficiency. Round trip efficiency is the efficiency of the battery when charging and discharging.
            minimum_state_of_charge: Minimum state of charge. Minimum state of charge is the minimum state of charge of the battery.
            maximum_state_of_charge: Maximum state of charge. Maximum state of charge is the maximum state of charge of the battery.
            timestep_in_hours: Timestep in hours. Timestep is the time step of the optimization.
            days: Number of days. Number of days is the number of days of the optimization.

        Returns:
            JSON string with the optimization results.
        """
        try:
            import pandas as pd
            from pyomo.environ import (
                ConcreteModel,
                ConstraintList,
                NonNegativeReals,
                Objective,
                Var,
                maximize,
            )
            from pyomo.opt import SolverFactory, TerminationCondition

            df = self._load_csv(csv_path)
            if energy_price_column not in df.columns:
                return json.dumps({"error": f"Column '{energy_price_column}' not found in CSV."})

            energy_price = df[energy_price_column].values
            horizon = int(days * 24 / timestep_in_hours)

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

            model.constraints.add(model.storage_soc[0] == 0.5 * battery_size_mwh)
            model.constraints.add(model.storage_soc[0] == model.storage_soc[horizon - 1])

            def total_cost(m):
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
                np.array(total_revenue_series) - charging_cost_series - np.array(degradation_penalty_series)
            )
            net_revenue = float(np.sum(net_revenue_series))

            state_of_charge = [model.storage_soc[t].value / battery_size_mwh for t in range(horizon)]

            output_rows = []
            for idx in range(horizon):
                output_rows.append(
                    {
                        "operation_profile": round(operation_profile[idx], 4),
                        "state_of_charge": round(state_of_charge[idx], 4),
                        "net_revenue": round(float(net_revenue_series[idx]), 4),
                        "total_revenue": round(float(total_revenue_series[idx]), 4),
                        "charging_cost": round(float(charging_cost_series[idx]), 4),
                        "degradation_penalty": round(float(degradation_penalty_series[idx]), 4),
                    }
                )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv_path = f"battery_revenue_optimization_{timestamp}.csv"
            pd.DataFrame(output_rows).to_csv(save_csv_path, index=False)

            return json.dumps(
                {
                    "run_description": run_description,
                    "net_revenue": round(net_revenue, 2),
                    "total_revenue": round(total_revenue, 2),
                    "charging_cost": round(charging_cost, 2),
                    "degradation_penalty": round(degradation_penalty, 2),
                    "rows": output_rows,
                    "saved_csv": save_csv_path,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Battery revenue optimization failed: {e}")
            return json.dumps({"error": str(e)})
