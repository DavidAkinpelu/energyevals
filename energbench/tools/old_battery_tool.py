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

        self.register_method("optimize_battery_arbitrage", self.optimize_arbitrage)
        self.register_method("calculate_battery_revenue", self.calculate_revenue)
        self.register_method("battery_revenue_optimization", self.battery_revenue_optimization)
        self.register_method("ercot_battery_revenue_optimization", self.ercot_battery_revenue_optimization)
        self.register_method("battery_optimization_bids", self.battery_optimization_bids)
        self.register_method("calculate_battery_project_irr", self.calculate_battery_project_irr)

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="optimize_battery_arbitrage",
                description=(
                    "Optimize battery dispatch for energy arbitrage given hourly prices. "
                    "Uses linear programming to maximize revenue from buying low and "
                    "selling high while respecting battery constraints."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "prices": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Hourly energy prices in $/MWh",
                        },
                        "battery_mw": {
                            "type": "number",
                            "description": "Battery power capacity in MW",
                        },
                        "battery_hours": {
                            "type": "number",
                            "description": "Battery duration in hours (energy = MW * hours)",
                        },
                        "efficiency": {
                            "type": "number",
                            "description": "Round-trip efficiency (default: 0.85)",
                            "default": 0.85,
                        },
                        "min_soc": {
                            "type": "number",
                            "description": "Minimum state of charge as fraction (default: 0.1)",
                            "default": 0.1,
                        },
                        "max_soc": {
                            "type": "number",
                            "description": "Maximum state of charge as fraction (default: 0.9)",
                            "default": 0.9,
                        },
                        "degradation_cost": {
                            "type": "number",
                            "description": "Degradation cost per MWh cycled in $ (default: 0)",
                            "default": 0,
                        },
                    },
                    "required": ["prices", "battery_mw", "battery_hours"],
                },
            ),
            ToolDefinition(
                name="calculate_battery_revenue",
                description=(
                    "Calculate revenue from a battery dispatch schedule. "
                    "Positive dispatch = discharging (selling), negative = charging (buying)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "prices": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Hourly energy prices in $/MWh",
                        },
                        "dispatch_mw": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Hourly dispatch in MW (positive=discharge, negative=charge)",
                        },
                    },
                    "required": ["prices", "dispatch_mw"],
                },
            ),
            ToolDefinition(
                name="battery_revenue_optimization",
                description=(
                    "Optimize battery revenue from energy arbitrage using Pyomo. "
                    "Reads a CSV of energy prices and returns revenue metrics and dispatch."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "run_description": {"type": "string"},
                        "csv_path": {"type": "string"},
                        "energy_price_column": {"type": "string"},
                        "battery_size_mw": {"type": "number"},
                        "battery_duration": {"type": "number"},
                        "battery_degradation_cost": {"type": "number", "default": 24.0},
                        "round_trip_efficiency": {"type": "number", "default": 0.81},
                        "minimum_state_of_charge": {"type": "number", "default": 0.1},
                        "maximum_state_of_charge": {"type": "number", "default": 0.8},
                        "timestep_in_hours": {"type": "number", "default": 1.0},
                        "days": {"type": "number", "default": 365.0},
                        "save_csv_path": {"type": "string"},
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
            ToolDefinition(
                name="ercot_battery_revenue_optimization",
                description=(
                    "Optimize ERCOT battery revenue stacking across energy and ancillary services."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "csv_path": {"type": "string"},
                        "run_description": {"type": "string"},
                        "energy_price_column": {"type": "string"},
                        "regup_price_column": {"type": "string"},
                        "regdown_price_column": {"type": "string"},
                        "rrs_price_column": {"type": "string"},
                        "nspin_price_column": {"type": "string"},
                        "ecrs_price_column": {"type": "string"},
                        "battery_size_mw": {"type": "number"},
                        "battery_duration": {"type": "number"},
                        "battery_degradation_cost": {"type": "number", "default": 24.0},
                        "round_trip_efficiency": {"type": "number", "default": 0.81},
                        "minimum_state_of_charge": {"type": "number", "default": 0.1},
                        "maximum_state_of_charge": {"type": "number", "default": 0.8},
                        "timestep_in_hours": {"type": "number", "default": 1.0},
                        "days": {"type": "number", "default": 365.0},
                        "save_csv_path": {"type": "string"},
                    },
                    "required": [
                        "csv_path",
                        "run_description",
                        "energy_price_column",
                        "regup_price_column",
                        "regdown_price_column",
                        "rrs_price_column",
                        "nspin_price_column",
                        "ecrs_price_column",
                        "battery_size_mw",
                        "battery_duration",
                    ],
                },
            ),
            ToolDefinition(
                name="battery_optimization_bids",
                description=(
                    "Generate battery bid curves using price scenarios and uncertainty."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "csv_path": {"type": "string"},
                        "run_description": {"type": "string"},
                        "scenario_probabilities": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "bid_price_margin": {"type": "number"},
                        "battery_mw": {"type": "number"},
                        "battery_duration": {"type": "number"},
                        "battery_efficiency": {"type": "number"},
                        "start_year": {"type": "integer"},
                        "start_month": {"type": "integer"},
                        "start_day": {"type": "integer"},
                        "resolution": {"type": "number"},
                        "terminal_soc": {"type": "number", "default": 0.5},
                        "degradation_cost": {"type": "number", "default": 24.0},
                        "save_csv_path": {"type": "string"},
                    },
                    "required": [
                        "csv_path",
                        "run_description",
                        "scenario_probabilities",
                        "bid_price_margin",
                        "battery_mw",
                        "battery_duration",
                        "battery_efficiency",
                        "start_year",
                        "start_month",
                        "start_day",
                        "resolution",
                    ],
                },
            ),
            ToolDefinition(
                name="calculate_battery_project_irr",
                description="Calculate project IRR for a battery storage asset.",
                parameters={
                    "type": "object",
                    "properties": {
                        "years": {"type": "integer"},
                        "capex": {"type": "number"},
                        "annual_revenue": {"type": "array", "items": {"type": "number"}},
                        "annual_opex": {"type": "array", "items": {"type": "number"}},
                        "tax_rate": {"type": "number", "default": 0.0},
                        "debt_amount": {"type": "number", "default": 0.0},
                        "debt_interest_rate": {"type": "number", "default": 0.0},
                        "debt_term": {"type": "integer", "default": 0},
                        "levered": {"type": "boolean", "default": False},
                        "apply_tax": {"type": "boolean", "default": False},
                    },
                    "required": ["years", "capex", "annual_revenue", "annual_opex"],
                },
            ),
        ]

    def optimize_arbitrage(
        self,
        prices: list[float],
        battery_mw: float,
        battery_hours: float,
        efficiency: float = 0.85,
        min_soc: float = 0.1,
        max_soc: float = 0.9,
        degradation_cost: float = 0,
    ) -> str:
        """Optimize battery dispatch for energy arbitrage."""
        try:
            prices = np.array(prices)
            n_hours = len(prices)
            battery_mwh = battery_mw * battery_hours

            charge_eff = np.sqrt(efficiency)
            discharge_eff = np.sqrt(efficiency)

            dispatch = np.zeros(n_hours)
            soc = np.ones(n_hours + 1) * 0.5 * battery_mwh

            sorted_hours = np.argsort(prices)
            low_price_hours = sorted_hours[: n_hours // 4]
            high_price_hours = sorted_hours[-(n_hours // 4) :]

            for t in range(n_hours):
                if t in low_price_hours:
                    max_charge = min(
                        battery_mw,
                        (max_soc * battery_mwh - soc[t]) / charge_eff,
                    )
                    dispatch[t] = -max_charge * 0.8
                elif t in high_price_hours:
                    max_discharge = min(
                        battery_mw,
                        (soc[t] - min_soc * battery_mwh) * discharge_eff,
                    )
                    dispatch[t] = max_discharge * 0.8

                if dispatch[t] < 0:
                    soc[t + 1] = soc[t] - dispatch[t] * charge_eff
                else:
                    soc[t + 1] = soc[t] - dispatch[t] / discharge_eff

                soc[t + 1] = np.clip(soc[t + 1], min_soc * battery_mwh, max_soc * battery_mwh)

            revenue = np.sum(dispatch * prices)
            total_discharge = np.sum(dispatch[dispatch > 0])
            total_charge = -np.sum(dispatch[dispatch < 0])
            cycles = total_discharge / battery_mwh
            degradation = cycles * degradation_cost * battery_mwh
            net_revenue = revenue - degradation

            avg_buy_price = (
                np.average(prices, weights=-dispatch * (dispatch < 0)) if np.any(dispatch < 0) else 0
            )
            avg_sell_price = (
                np.average(prices, weights=dispatch * (dispatch > 0)) if np.any(dispatch > 0) else 0
            )

            result = {
                "battery_mw": battery_mw,
                "battery_mwh": battery_mwh,
                "efficiency": efficiency,
                "num_hours": n_hours,
                "revenue": {
                    "gross_revenue": round(revenue, 2),
                    "degradation_cost": round(degradation, 2),
                    "net_revenue": round(net_revenue, 2),
                },
                "operations": {
                    "total_discharge_mwh": round(total_discharge, 2),
                    "total_charge_mwh": round(total_charge, 2),
                    "cycles": round(cycles, 2),
                    "avg_buy_price": round(avg_buy_price, 2),
                    "avg_sell_price": round(avg_sell_price, 2),
                    "spread": round(avg_sell_price - avg_buy_price, 2),
                },
                "dispatch_sample": dispatch[:24].tolist(),
                "soc_sample": soc[:25].tolist(),
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Battery optimization failed: {e}")
            return json.dumps({"error": str(e)})

    def calculate_revenue(
        self,
        prices: list[float],
        dispatch_mw: list[float],
    ) -> str:
        """Calculate revenue from a dispatch schedule."""
        try:
            prices = np.array(prices)
            dispatch = np.array(dispatch_mw)

            if len(prices) != len(dispatch):
                return json.dumps(
                    {
                        "error": "prices and dispatch_mw must have same length",
                        "prices_length": len(prices),
                        "dispatch_length": len(dispatch),
                    }
                )

            discharge_mask = dispatch > 0
            charge_mask = dispatch < 0

            discharge_revenue = np.sum(dispatch[discharge_mask] * prices[discharge_mask])
            charge_cost = -np.sum(dispatch[charge_mask] * prices[charge_mask])
            net_revenue = discharge_revenue - charge_cost

            total_discharge = np.sum(dispatch[discharge_mask])
            total_charge = -np.sum(dispatch[charge_mask])

            result = {
                "discharge_revenue": round(discharge_revenue, 2),
                "charge_cost": round(charge_cost, 2),
                "net_revenue": round(net_revenue, 2),
                "total_discharge_mwh": round(total_discharge, 2),
                "total_charge_mwh": round(total_charge, 2),
                "avg_discharge_price": (
                    round(discharge_revenue / total_discharge, 2) if total_discharge > 0 else 0
                ),
                "avg_charge_price": (
                    round(charge_cost / total_charge, 2) if total_charge > 0 else 0
                ),
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Revenue calculation failed: {e}")
            return json.dumps({"error": str(e)})

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
        save_csv_path: Optional[str] = None,
    ) -> str:
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
            from pyomo.opt import SolverFactory

            df = self._load_csv(csv_path)
            if energy_price_column not in df.columns:
                return json.dumps({"error": f"Column '{energy_price_column}' not found in CSV."})

            energy_price = df[energy_price_column].values
            horizon = int(days * 24 / timestep_in_hours)
            horizon = min(horizon, len(energy_price))

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
            solver.solve(model, report_timing=True)

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

            if save_csv_path:
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

    def ercot_battery_revenue_optimization(
        self,
        csv_path: str,
        run_description: str,
        energy_price_column: str,
        regup_price_column: str,
        regdown_price_column: str,
        rrs_price_column: str,
        nspin_price_column: str,
        ecrs_price_column: str,
        battery_size_mw: float,
        battery_duration: float,
        battery_degradation_cost: float = 24.0,
        round_trip_efficiency: float = 0.81,
        minimum_state_of_charge: float = 0.1,
        maximum_state_of_charge: float = 0.8,
        timestep_in_hours: float = 1.0,
        days: float = 365.0,
        save_csv_path: Optional[str] = None,
    ) -> str:
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
            from pyomo.opt import SolverFactory

            df = self._load_csv(csv_path)
            required_columns = [
                energy_price_column,
                regup_price_column,
                regdown_price_column,
                rrs_price_column,
                nspin_price_column,
                ecrs_price_column,
            ]
            for col in required_columns:
                if col not in df.columns:
                    return json.dumps({"error": f"Column '{col}' not found in CSV."})

            energy_price = df[energy_price_column].values
            regup_price = df[regup_price_column].values
            regdown_price = df[regdown_price_column].values
            rrs_price = df[rrs_price_column].values
            nspin_price = df[nspin_price_column].values
            ecrs_price = df[ecrs_price_column].values

            horizon = int(days * 24 / timestep_in_hours)
            horizon = min(horizon, len(energy_price))

            battery_size_mwh = battery_size_mw * battery_duration
            charge_eff = np.sqrt(round_trip_efficiency)
            discharge_eff = np.sqrt(round_trip_efficiency)

            model = ConcreteModel()
            model.storage_soc = Var(range(horizon), domain=NonNegativeReals)
            model.charge_power = Var(range(horizon), domain=NonNegativeReals)
            model.discharge_power = Var(range(horizon), domain=NonNegativeReals)
            model.abs_power = Var(range(horizon), domain=NonNegativeReals)

            model.charge_power_arbitrage = Var(range(horizon), domain=NonNegativeReals)
            model.discharge_power_arbitrage = Var(range(horizon), domain=NonNegativeReals)

            model.regup_capacity = Var(range(horizon), domain=NonNegativeReals)
            model.regdown_capacity = Var(range(horizon), domain=NonNegativeReals)
            model.rrs_capacity = Var(range(horizon), domain=NonNegativeReals)
            model.nspin_capacity = Var(range(horizon), domain=NonNegativeReals)
            model.ecrs_capacity = Var(range(horizon), domain=NonNegativeReals)

            reg_duration = 1
            rrs_duration = 1
            nspin_duration = 4
            ecrs_duration = 2

            model.constraints = ConstraintList()
            for t in range(horizon):
                model.constraints.add(model.storage_soc[t] >= minimum_state_of_charge * battery_size_mwh)
                model.constraints.add(model.storage_soc[t] <= maximum_state_of_charge * battery_size_mwh)
                model.constraints.add(model.charge_power[t] <= battery_size_mw)
                model.constraints.add(model.discharge_power[t] <= battery_size_mw)

                model.constraints.add(model.regup_capacity[t] <= battery_size_mw)
                model.constraints.add(model.regdown_capacity[t] <= battery_size_mw)
                model.constraints.add(model.rrs_capacity[t] <= battery_size_mw)
                model.constraints.add(model.nspin_capacity[t] <= battery_size_mw)
                model.constraints.add(model.ecrs_capacity[t] <= battery_size_mw)

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

            for t in range(horizon):
                if t > 0:
                    model.constraints.add(
                        model.storage_soc[t - 1] - minimum_state_of_charge * battery_size_mwh
                        >= model.discharge_power_arbitrage[t]
                        + model.regup_capacity[t] * reg_duration
                        + model.rrs_capacity[t] * rrs_duration
                        + model.nspin_capacity[t] * nspin_duration
                        + model.ecrs_capacity[t] * ecrs_duration
                    )
                    model.constraints.add(
                        maximum_state_of_charge * battery_size_mwh - model.storage_soc[t - 1]
                        >= model.charge_power_arbitrage[t] + model.regdown_capacity[t] * reg_duration
                    )

                model.constraints.add(
                    model.charge_power[t] == model.charge_power_arbitrage[t] + model.regdown_capacity[t]
                )
                model.constraints.add(
                    model.discharge_power[t]
                    == model.discharge_power_arbitrage[t]
                    + model.regup_capacity[t]
                    + model.rrs_capacity[t]
                    + model.nspin_capacity[t]
                    + model.ecrs_capacity[t]
                )

            def total_cost(m):
                arbitrage_revenue = sum(
                    (timestep_in_hours * energy_price[t] * m.discharge_power_arbitrage[t]) for t in range(horizon)
                )
                regup_revenue = sum(
                    (timestep_in_hours * regup_price[t] * m.regup_capacity[t]) for t in range(horizon)
                )
                regdown_revenue = sum(
                    (timestep_in_hours * regdown_price[t] * m.regdown_capacity[t]) for t in range(horizon)
                )
                rrs_revenue = sum((timestep_in_hours * rrs_price[t] * m.rrs_capacity[t]) for t in range(horizon))
                nspin_revenue = sum(
                    (timestep_in_hours * nspin_price[t] * m.nspin_capacity[t]) for t in range(horizon)
                )
                ecrs_revenue = sum(
                    (timestep_in_hours * ecrs_price[t] * m.ecrs_capacity[t]) for t in range(horizon)
                )
                charge_cost = sum((timestep_in_hours * energy_price[t] * m.charge_power[t]) for t in range(horizon))
                degradation_penalty = sum(
                    (timestep_in_hours * battery_degradation_cost * m.abs_power[t]) for t in range(horizon)
                )
                return (
                    arbitrage_revenue
                    + regup_revenue
                    + regdown_revenue
                    + rrs_revenue
                    + nspin_revenue
                    + ecrs_revenue
                    - charge_cost
                    - degradation_penalty
                )

            model.cost = Objective(rule=total_cost, sense=maximize)

            solver = SolverFactory("ipopt")
            if solver is None or not solver.available():
                return json.dumps({"error": "IPOPT solver not available. Install it to run optimization."})
            solver.solve(model, report_timing=True)

            arbitrage_revenue_series = [
                timestep_in_hours * energy_price[t] * model.discharge_power_arbitrage[t].value
                for t in range(horizon)
            ]
            regup_revenue_series = [
                timestep_in_hours * regup_price[t] * model.regup_capacity[t].value for t in range(horizon)
            ]
            regdown_revenue_series = [
                timestep_in_hours * regdown_price[t] * model.regdown_capacity[t].value for t in range(horizon)
            ]
            rrs_revenue_series = [
                timestep_in_hours * rrs_price[t] * model.rrs_capacity[t].value for t in range(horizon)
            ]
            nspin_revenue_series = [
                timestep_in_hours * nspin_price[t] * model.nspin_capacity[t].value for t in range(horizon)
            ]
            ecrs_revenue_series = [
                timestep_in_hours * ecrs_price[t] * model.ecrs_capacity[t].value for t in range(horizon)
            ]

            total_revenue_series = (
                np.array(arbitrage_revenue_series)
                + np.array(regup_revenue_series)
                + np.array(regdown_revenue_series)
                + np.array(rrs_revenue_series)
                + np.array(nspin_revenue_series)
                + np.array(ecrs_revenue_series)
            )
            degradation_penalty_series = [
                -1 * timestep_in_hours * battery_degradation_cost * model.abs_power[t].value for t in range(horizon)
            ]
            operation_profile = [
                model.discharge_power[t].value - model.charge_power[t].value for t in range(horizon)
            ]
            charging_only = np.minimum(0, np.array(operation_profile))
            charging_cost_series = timestep_in_hours * (charging_only * energy_price[: len(charging_only)])

            net_revenue_series = total_revenue_series - charging_cost_series - np.array(degradation_penalty_series)

            state_of_charge = [model.storage_soc[t].value / battery_size_mwh for t in range(horizon)]

            rows = []
            for idx in range(horizon):
                rows.append(
                    {
                        "net_revenue": round(float(net_revenue_series[idx]), 4),
                        "arbitrage_revenue": round(float(arbitrage_revenue_series[idx]), 4),
                        "regup_revenue": round(float(regup_revenue_series[idx]), 4),
                        "regdown_revenue": round(float(regdown_revenue_series[idx]), 4),
                        "rrs_revenue": round(float(rrs_revenue_series[idx]), 4),
                        "nspin_revenue": round(float(nspin_revenue_series[idx]), 4),
                        "ecrs_revenue": round(float(ecrs_revenue_series[idx]), 4),
                        "total_revenue": round(float(total_revenue_series[idx]), 4),
                        "charging_cost": round(float(charging_cost_series[idx]), 4),
                        "degradation_penalty": round(float(degradation_penalty_series[idx]), 4),
                        "operation_profile": round(float(operation_profile[idx]), 4),
                        "state_of_charge": round(float(state_of_charge[idx]), 4),
                        "regup_capacity": round(float(model.regup_capacity[idx].value), 4),
                        "regdown_capacity": round(float(model.regdown_capacity[idx].value), 4),
                        "rrs_capacity": round(float(model.rrs_capacity[idx].value), 4),
                        "nspin_capacity": round(float(model.nspin_capacity[idx].value), 4),
                        "ecrs_capacity": round(float(model.ecrs_capacity[idx].value), 4),
                        "energy_only_capacity": round(float(operation_profile[idx]), 4),
                    }
                )

            if save_csv_path:
                pd.DataFrame(rows).to_csv(save_csv_path, index=False)

            return json.dumps(
                {
                    "run_description": run_description,
                    "net_revenue": round(float(np.sum(net_revenue_series)), 2),
                    "total_revenue": round(float(np.sum(total_revenue_series)), 2),
                    "charging_cost": round(float(np.sum(charging_cost_series)), 2),
                    "degradation_penalty": round(float(np.sum(degradation_penalty_series)), 2),
                    "rows": rows,
                    "saved_csv": save_csv_path,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"ERCOT battery optimization failed: {e}")
            return json.dumps({"error": str(e)})

    def battery_optimization_bids(
        self,
        csv_path: str,
        run_description: str,
        scenario_probabilities: list[float],
        bid_price_margin: float,
        battery_mw: float,
        battery_duration: float,
        battery_efficiency: float,
        start_year: int,
        start_month: int,
        start_day: int,
        resolution: float,
        terminal_soc: float = 0.5,
        degradation_cost: float = 24.0,
        save_csv_path: Optional[str] = None,
    ) -> str:
        try:
            import pandas as pd
            from datetime import timedelta

            try:
                from batteries_included.model.common import Battery, PriceScenarios, Scenario, TimeSeries
                from batteries_included.model.optimization import ModelBuilder
            except ImportError as e:
                return json.dumps(
                    {
                        "error": "batteries_included not installed. Install batteries-included to run this tool.",
                        "detail": str(e),
                    }
                )

            prices = self._load_csv(csv_path)
            prices = prices.drop(
                columns=prices.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]", "object", "int"])
            )

            def build_scenarios(df, probs, start, resolution_td, axis: str = "columns"):
                if axis not in ("columns", "index"):
                    raise ValueError('axis must be "columns" or "index"')
                if axis == "columns":
                    scenario_ids = list(df.columns)
                    series_iter = [df[col].tolist() for col in df.columns]
                else:
                    scenario_ids = list(df.index)
                    series_iter = [df.loc[idx].tolist() for idx in df.index]
                probs_arr = np.asarray(probs, dtype=float)
                if probs_arr.size != len(scenario_ids):
                    raise ValueError("Length of probabilities must match number of scenarios")
                if (probs_arr < 0).any():
                    raise ValueError("Probabilities must be non-negative")
                total_prob = probs_arr.sum()
                if not np.isclose(total_prob, 1.0):
                    probs_arr = probs_arr / total_prob
                keys = [f"Scenario {i+1}" for i in range(len(scenario_ids))]
                scenarios = {}
                for name, p, vals in zip(keys, probs_arr, series_iter):
                    scenarios[name] = Scenario(
                        probability=float(p),
                        value=TimeSeries(start=start, resolution=resolution_td, values=list(vals)),
                    )
                return scenarios

            start = datetime(start_year, start_month, start_day)
            resolution_td = timedelta(hours=resolution)
            price_scenarios = build_scenarios(prices, scenario_probabilities, start, resolution_td, axis="columns")
            price_scenarios = PriceScenarios(scenarios=price_scenarios)

            battery = Battery(
                duration=timedelta(hours=battery_duration),
                power=battery_mw,
                efficiency=battery_efficiency,
                variable_cost=degradation_cost,
            )

            solution = (
                ModelBuilder(battery=battery, price_scenarios=price_scenarios)
                .constrain_storage_level(soc_start=("==", terminal_soc), soc_end=("==", terminal_soc))
                .constrain_bidding_strategy()
                .constrain_imbalance()
                .constrain_simultanous_dispatch()
                .add_objective(penalize_imbalance=100000.0)
                .solve()
            )

            bids_df = pd.DataFrame(
                solution.collect_bids(
                    margin=bid_price_margin,
                    remove_quantity_bids_below=0.0,
                )
            ).round(2)

            if save_csv_path:
                bids_df.to_csv(save_csv_path, index=False)

            return json.dumps(
                {
                    "run_description": run_description,
                    "rows": bids_df.to_dict(orient="records"),
                    "saved_csv": save_csv_path,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Battery bid optimization failed: {e}")
            return json.dumps({"error": str(e)})

    def calculate_battery_project_irr(
        self,
        years: int,
        capex: float,
        annual_revenue: list[float],
        annual_opex: list[float],
        tax_rate: float = 0.0,
        debt_amount: float = 0.0,
        debt_interest_rate: float = 0.0,
        debt_term: int = 0,
        levered: bool = False,
        apply_tax: bool = False,
    ) -> str:
        try:
            import numpy as np
            import pandas as pd

            if len(annual_revenue) != years or len(annual_opex) != years:
                return json.dumps(
                    {"error": "Length of annual_revenue and annual_opex must match the number of years."}
                )

            cash_flows = []
            initial_investment = -capex
            if levered:
                equity_contribution = capex - debt_amount
                cash_flows.append(-equity_contribution)
            else:
                cash_flows.append(initial_investment)

            annual_debt_payment_schedule = []
            if levered and debt_amount > 0 and debt_term > 0:
                annual_principal = debt_amount / debt_term
                for i in range(years):
                    if i < debt_term:
                        interest = (debt_amount - i * annual_principal) * debt_interest_rate
                        payment = interest + annual_principal
                    else:
                        payment = 0
                    annual_debt_payment_schedule.append(payment)
            else:
                annual_debt_payment_schedule = [0] * years

            for year in range(years):
                revenue = annual_revenue[year]
                opex = annual_opex[year]
                debt_payment = annual_debt_payment_schedule[year] if levered else 0

                ebt = revenue - opex - debt_payment
                tax = tax_rate * ebt if apply_tax and ebt > 0 else 0
                net_cash = ebt - tax
                cash_flows.append(net_cash)

            irr_value = float(np.irr(cash_flows))
            cash_flow_df = pd.DataFrame({"cash_flows": cash_flows})
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.abspath(f"cash_flows_{timestamp}.csv")
            cash_flow_df.to_csv(save_path, index=False)

            return json.dumps(
                {
                    "irr": irr_value,
                    "cash_flows": cash_flows,
                    "saved_csv": save_path,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"IRR calculation failed: {e}")
            return json.dumps({"error": str(e)})
