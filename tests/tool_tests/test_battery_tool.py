import csv
import json

import pandas as pd
import pytest

from energbench.tools.battery_tool import BatteryOptimizationTool


@pytest.fixture
def price_csv(tmp_path):
    """Create a sample price CSV for testing."""
    csv_path = tmp_path / "prices.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["price"])
        for hour in range(24):
            writer.writerow([20 + hour])  # Prices from 20 to 43
    return str(csv_path)


class TestBatteryOptimizationToolUnit:
    """Unit tests for BatteryOptimizationTool."""

    def test_init(self):
        """Test tool initialization."""
        tool = BatteryOptimizationTool()
        assert tool.name == "battery_optimization"

    def test_get_tools_definition(self):
        """Test tool definitions are properly structured."""
        tool = BatteryOptimizationTool()
        tools = tool.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "battery_revenue_optimization"
        assert "battery" in tools[0].description.lower()

        required = tools[0].parameters["required"]
        assert "csv_path" in required
        assert "energy_price_column" in required
        assert "battery_size_mw" in required
        assert "battery_duration" in required

    @pytest.mark.asyncio
    async def test_optimization_success(self, price_csv):
        """Test successful optimization with valid inputs."""
        tool = BatteryOptimizationTool()

        result = tool.battery_revenue_optimization(
            run_description="test",
            csv_path=price_csv,
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

        result_data = json.loads(result)
        assert "revenue" in result_data or "error" not in result_data

    @pytest.mark.asyncio
    async def test_optimization_missing_csv(self):
        """Test optimization with non-existent CSV file."""
        tool = BatteryOptimizationTool()

        result = tool.battery_revenue_optimization(
            run_description="test",
            csv_path="/nonexistent/file.csv",
            energy_price_column="price",
            battery_size_mw=1.0,
            battery_duration=2.0,
        )

        result_data = json.loads(result)
        assert "error" in result_data

    @pytest.mark.asyncio
    async def test_optimization_missing_column(self, price_csv):
        """Test optimization with incorrect column name."""
        tool = BatteryOptimizationTool()

        result = tool.battery_revenue_optimization(
            run_description="test",
            csv_path=price_csv,
            energy_price_column="nonexistent_column",
            battery_size_mw=1.0,
            battery_duration=2.0,
        )

        result_data = json.loads(result)
        assert "error" in result_data
        assert "not found" in result_data["error"].lower()

    @pytest.mark.asyncio
    async def test_load_csv_helper(self, price_csv):
        """Test the _load_csv helper method."""
        tool = BatteryOptimizationTool()
        df = tool._load_csv(price_csv)

        assert isinstance(df, pd.DataFrame)
        assert "price" in df.columns
        assert len(df) == 24

    @pytest.mark.asyncio
    async def test_load_csv_file_not_found(self):
        """Test _load_csv with non-existent file."""
        tool = BatteryOptimizationTool()

        with pytest.raises(FileNotFoundError):
            tool._load_csv("/nonexistent/file.csv")


@pytest.mark.integration
@pytest.mark.slow
class TestBatteryOptimizationToolIntegration:
    """Integration tests with full optimization solver."""

    @pytest.mark.asyncio
    async def test_full_optimization_24_hours(self, tmp_path):
        """Test full optimization over 24 hours."""
        csv_path = tmp_path / "prices.csv"
        prices = [20, 25, 30, 35, 40, 38, 36, 34, 32, 30, 28, 26,
                  24, 22, 21, 23, 25, 28, 31, 35, 38, 36, 33, 28]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["energy_price"])
            for price in prices:
                writer.writerow([price])

        tool = BatteryOptimizationTool()
        result = tool.battery_revenue_optimization(
            run_description="full_test",
            csv_path=str(csv_path),
            energy_price_column="energy_price",
            battery_size_mw=2.0,
            battery_duration=4.0,
            battery_degradation_cost=10.0,
            round_trip_efficiency=0.85,
            minimum_state_of_charge=0.2,
            maximum_state_of_charge=0.9,
            timestep_in_hours=1.0,
            days=1.0,
        )

        result_data = json.loads(result)
        assert "error" not in result_data
        assert "revenue" in result_data or "total_revenue" in str(result_data).lower()
