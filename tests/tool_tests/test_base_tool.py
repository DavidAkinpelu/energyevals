from typing import Literal

import pytest

from energbench.tools.base_tool import BaseTool, ToolRegistry, ToolResult, tool_method


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_tool_result_success(self):
        """Test creating a successful tool result."""
        result = ToolResult(
            success=True,
            data={"value": 42},
            error=None,
        )

        assert result.success
        assert result.data["value"] == 42
        assert result.error is None

    def test_tool_result_failure(self):
        """Test creating a failed tool result."""
        result = ToolResult(
            success=False,
            data=None,
            error="Something went wrong",
        )

        assert not result.success
        assert result.error == "Something went wrong"
        assert result.data is None

    def test_tool_result_to_json(self):
        """Test converting tool result to JSON."""
        result = ToolResult(
            success=True,
            data={"value": 42, "name": "test"},
            error=None,
            metadata={"execution_time": 0.5},
        )

        json_str = result.to_json()

        assert "value" in json_str
        assert "42" in json_str
        assert "test" in json_str
        assert "execution_time" in json_str


class TestBaseTool:
    """Tests for BaseTool class."""

    def test_base_tool_initialization(self):
        """Test BaseTool initialization."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="test_tool", description="A test tool")

        tool = TestTool()

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert len(tool._methods) == 0

    def test_tool_method_decorator(self):
        """Test that @tool_method auto-registers a method."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="test_tool", description="Test")

            @tool_method(parameters={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            })
            def test_method(self, input: str) -> str:
                """Process the input."""
                return f"Processed: {input}"

        tool = TestTool()

        assert "test_method" in tool._methods
        assert callable(tool._methods["test_method"])

    def test_get_tools_from_decorator(self):
        """Test that get_tools() builds definitions from @tool_method."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="test_tool", description="Test")

            @tool_method(parameters={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            })
            def test_method(self, input: str) -> str:
                """Process the input."""
                return f"Processed: {input}"

        tool = TestTool()
        tools = tool.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "test_method"
        assert tools[0].description == "Process the input."

    @pytest.mark.asyncio
    async def test_execute_sync_method(self):
        """Test executing a synchronous method."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="test_tool", description="Test")

            @tool_method(parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            })
            def sync_method(self, value: int) -> str:
                """Double a value."""
                return f"Result: {value * 2}"

        tool = TestTool()
        result = await tool.execute("sync_method", value=21)

        assert result.success
        assert "42" in result.data

    @pytest.mark.asyncio
    async def test_execute_async_method(self):
        """Test executing an asynchronous method."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="test_tool", description="Test")

            @tool_method(parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            })
            async def async_method(self, value: str) -> str:
                """Return an async result."""
                return f"Async result: {value}"

        tool = TestTool()
        result = await tool.execute("async_method", value="test")

        assert result.success
        assert "Async result: test" in result.data

    @pytest.mark.asyncio
    async def test_execute_nonexistent_method(self):
        """Test executing a method that doesn't exist."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="test_tool", description="Test")

        tool = TestTool()
        result = await tool.execute("nonexistent", arg="value")

        assert not result.success
        assert result.error is not None
        assert "unknown method" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_method_with_exception(self):
        """Test executing a method that raises an exception."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="test_tool", description="Test")

            @tool_method(parameters={"type": "object", "properties": {}})
            def error_method(self) -> str:
                """Raise an error."""
                raise ValueError("Intentional error")

        tool = TestTool()
        result = await tool.execute("error_method")

        assert not result.success
        assert result.error is not None
        assert "Intentional error" in result.error


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = ToolRegistry()

        assert len(registry.get_all_tools()) == 0

    def test_register_tool(self):
        """Test registering a tool."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="test_tool", description="Test")

            @tool_method(parameters={"type": "object", "properties": {}})
            def test_method(self) -> str:
                """A test method."""
                return "ok"

        registry = ToolRegistry()
        tool = TestTool()

        registry.register(tool)

        all_tools = registry.get_all_tools()
        assert len(all_tools) == 1
        assert all_tools[0].name == "test_method"

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""

        class Tool1(BaseTool):
            def __init__(self):
                super().__init__(name="tool1", description="First")

            @tool_method(parameters={})
            def method1(self) -> str:
                """First method."""
                return "1"

        class Tool2(BaseTool):
            def __init__(self):
                super().__init__(name="tool2", description="Second")

            @tool_method(parameters={})
            def method2(self) -> str:
                """Second method."""
                return "2"

        registry = ToolRegistry()
        registry.register(Tool1())
        registry.register(Tool2())

        all_tools = registry.get_all_tools()
        assert len(all_tools) == 2

        tool_names = {t.name for t in all_tools}
        assert "method1" in tool_names
        assert "method2" in tool_names

    @pytest.mark.asyncio
    async def test_execute_tool_from_registry(self):
        """Test executing a tool through the registry."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="calc_tool", description="Calculator")

            @tool_method(parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            })
            def add(self, a: int, b: int) -> int:
                """Add two numbers."""
                return a + b

        registry = ToolRegistry()
        registry.register(TestTool())

        result = await registry.execute("add", a=10, b=32)

        assert result.success
        assert result.data == 42

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist in registry."""
        registry = ToolRegistry()

        result = await registry.execute("nonexistent_tool", arg="value")

        assert not result.success
        assert result.error is not None

    def test_get_tool_by_name(self):
        """Test retrieving a specific tool definition by name."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="test", description="Test")

            @tool_method(parameters={})
            def find_me(self) -> str:
                """Target method."""
                return "found"

            @tool_method(parameters={})
            def other(self) -> str:
                """Other method."""
                return "other"

        registry = ToolRegistry()
        registry.register(TestTool())

        all_tools = registry.get_all_tools()
        target_tool = next((t for t in all_tools if t.name == "find_me"), None)

        assert target_tool is not None
        assert target_tool.description == "Target method."

    def test_tool_registry_with_multiple_methods_per_tool(self):
        """Test a tool with multiple registered methods."""

        class MultiMethodTool(BaseTool):
            def __init__(self):
                super().__init__(name="multi", description="Multi-method tool")

            @tool_method(parameters={})
            def method1(self) -> str:
                """First method."""
                return "result1"

            @tool_method(parameters={})
            def method2(self) -> str:
                """Second method."""
                return "result2"

        registry = ToolRegistry()
        registry.register(MultiMethodTool())

        all_tools = registry.get_all_tools()
        assert len(all_tools) == 2

        tool_names = {t.name for t in all_tools}
        assert "method1" in tool_names
        assert "method2" in tool_names


class TestAutoSchema:
    """Tests for automatic JSON Schema generation from type hints + docstrings."""

    def test_auto_schema_basic_types(self):
        """Test auto-generation for str, int, float, bool parameters."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(self, name: str, count: int, rate: float, flag: bool) -> str:
                """Do something.

                Args:
                    name: The name.
                    count: The count.
                    rate: The rate.
                    flag: A flag.
                """
                return ""

        tool = TestTool()
        tools = tool.get_tools()
        schema = tools[0].parameters

        assert schema["type"] == "object"
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["rate"]["type"] == "number"
        assert schema["properties"]["flag"]["type"] == "boolean"
        assert set(schema["required"]) == {"name", "count", "rate", "flag"}

    def test_auto_schema_optional_not_required(self):
        """Test that Optional params with defaults are not required."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(self, query: str, limit: int | None = None) -> str:
                """Search.

                Args:
                    query: The query.
                    limit: Max results.
                """
                return ""

        tool = TestTool()
        schema = tool.get_tools()[0].parameters

        assert schema["required"] == ["query"]
        assert "limit" in schema["properties"]
        assert schema["properties"]["limit"]["type"] == "integer"

    def test_auto_schema_defaults(self):
        """Test that default values are included in the schema."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(self, x: int = 10, flag: bool = True) -> str:
                """Do it.

                Args:
                    x: A number.
                    flag: A flag.
                """
                return ""

        tool = TestTool()
        schema = tool.get_tools()[0].parameters

        assert "required" not in schema
        assert schema["properties"]["x"]["default"] == 10
        assert schema["properties"]["flag"]["default"] is True

    def test_auto_schema_literal_enum(self):
        """Test that Literal type hints produce enum constraints."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(self, mode: Literal["fast", "slow"] = "fast") -> str:
                """Run.

                Args:
                    mode: Speed mode.
                """
                return ""

        tool = TestTool()
        schema = tool.get_tools()[0].parameters
        prop = schema["properties"]["mode"]

        assert prop["type"] == "string"
        assert prop["enum"] == ["fast", "slow"]
        assert prop["default"] == "fast"

    def test_auto_schema_literal_int_enum(self):
        """Test that Literal[int] produces integer enum."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(self, tracking: Literal[0, 1, 2] = 0) -> str:
                """Track.

                Args:
                    tracking: Tracking mode.
                """
                return ""

        tool = TestTool()
        prop = tool.get_tools()[0].parameters["properties"]["tracking"]

        assert prop["type"] == "integer"
        assert prop["enum"] == [0, 1, 2]

    def test_auto_schema_list_type(self):
        """Test that list[str] produces array schema."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(self, tags: list[str]) -> str:
                """Tag it.

                Args:
                    tags: List of tags.
                """
                return ""

        tool = TestTool()
        prop = tool.get_tools()[0].parameters["properties"]["tags"]

        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"

    def test_auto_schema_union_one_of(self):
        """Test that Union[bool, dict] produces oneOf schema."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(self, text: bool | dict = True) -> str:
                """Get text.

                Args:
                    text: Include text.
                """
                return ""

        tool = TestTool()
        prop = tool.get_tools()[0].parameters["properties"]["text"]

        assert "oneOf" in prop
        types = [s["type"] for s in prop["oneOf"]]
        assert "boolean" in types
        assert "object" in types

    def test_auto_schema_no_args(self):
        """Test that a no-arg method produces empty properties."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(self) -> str:
                """List everything."""
                return ""

        tool = TestTool()
        schema = tool.get_tools()[0].parameters

        assert schema == {"type": "object", "properties": {}}

    def test_auto_schema_docstring_descriptions(self):
        """Test that docstring Args descriptions are used."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(self, query: str, limit: int = 10) -> str:
                """Search things.

                Args:
                    query: The search query describing what you want.
                    limit: Maximum number of results to return.
                """
                return ""

        tool = TestTool()
        props = tool.get_tools()[0].parameters["properties"]

        assert props["query"]["description"] == "The search query describing what you want."
        assert props["limit"]["description"] == "Maximum number of results to return."

    def test_auto_schema_description_from_summary(self):
        """Test that tool description comes from docstring summary."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(self, x: str) -> str:
                """Search the web for relevant information.

                Args:
                    x: Input.
                """
                return ""

        tool = TestTool()
        td = tool.get_tools()[0]

        assert td.description == "Search the web for relevant information."

    def test_explicit_parameters_override(self):
        """Test that explicit parameters dict still works (backward compat)."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method(parameters={
                "type": "object",
                "properties": {"custom": {"type": "string"}},
                "required": ["custom"],
            })
            def method(self, custom: str) -> str:
                """Do it."""
                return ""

        tool = TestTool()
        schema = tool.get_tools()[0].parameters

        assert schema["properties"]["custom"]["type"] == "string"
        assert schema["required"] == ["custom"]

    def test_auto_schema_optional_literal(self):
        """Test Optional[Literal[...]] produces enum without being required."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(
                self,
                location: str,
                units: Literal["metric", "imperial"] | None = None,
            ) -> str:
                """Get weather.

                Args:
                    location: The location.
                    units: Units of measurement.
                """
                return ""

        tool = TestTool()
        schema = tool.get_tools()[0].parameters

        assert schema["required"] == ["location"]
        prop = schema["properties"]["units"]
        assert prop["type"] == "string"
        assert prop["enum"] == ["metric", "imperial"]

    def test_parse_docstring_args_parameters_section(self):
        """Test parsing docstrings that use 'Parameters:' instead of 'Args:'."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="t", description="T")

            @tool_method()
            def method(self, start_date: str, keyword: str | None = None) -> str:
                """Search filings.

                Parameters:
                    start_date: Start date in YYYY-MM-DD format.
                    keyword: Keyword to search.
                """
                return ""

        tool = TestTool()
        props = tool.get_tools()[0].parameters["properties"]

        assert props["start_date"]["description"] == "Start date in YYYY-MM-DD format."
        assert props["keyword"]["description"] == "Keyword to search."


class TestAutoSchemaRegression:
    """Regression tests verifying real tool classes produce correct schemas.

    Each test instantiates a production tool and asserts the auto-generated
    schema matches the previously hand-written JSON Schema in structure,
    types, required fields, enum constraints, and defaults.
    """

    def test_gridstatus_list_datasets_schema(self):
        from energbench.tools.gridstatus_tool import GridStatusAPITool

        tool = GridStatusAPITool()
        td = next(t for t in tool.get_tools() if t.name == "list_gridstatus_datasets")

        assert td.parameters == {"type": "object", "properties": {}}
        assert "datasets" in td.description.lower()

    def test_gridstatus_inspect_dataset_schema(self):
        from energbench.tools.gridstatus_tool import GridStatusAPITool

        tool = GridStatusAPITool()
        td = next(t for t in tool.get_tools() if t.name == "inspect_gridstatus_dataset")
        schema = td.parameters

        assert schema["required"] == ["dataset_id"]
        assert schema["properties"]["dataset_id"]["type"] == "string"

    def test_gridstatus_query_dataset_schema(self):
        from energbench.tools.gridstatus_tool import GridStatusAPITool

        tool = GridStatusAPITool()
        td = next(t for t in tool.get_tools() if t.name == "query_gridstatus_dataset")
        schema = td.parameters

        assert schema["required"] == ["dataset_id"]
        assert len(schema["properties"]) == 14

        assert schema["properties"]["dataset_id"]["type"] == "string"
        assert schema["properties"]["limit"]["type"] == "integer"
        assert schema["properties"]["columns"]["type"] == "array"
        assert schema["properties"]["columns"]["items"]["type"] == "string"

        rf = schema["properties"]["resample_function"]
        assert rf["type"] == "string"
        assert rf["enum"] == ["mean", "sum", "min", "max", "count", "stddev", "variance"]
        assert rf["default"] == "mean"


    def test_battery_optimization_schema(self):
        from energbench.tools.battery_tool import BatteryOptimizationTool

        tool = BatteryOptimizationTool()
        td = tool.get_tools()[0]
        schema = td.parameters

        assert set(schema["required"]) == {
            "run_description", "csv_path", "energy_price_column",
            "battery_size_mw", "battery_duration",
        }
        assert len(schema["properties"]) == 11

        assert schema["properties"]["battery_size_mw"]["type"] == "number"
        assert schema["properties"]["battery_duration"]["type"] == "number"
        assert schema["properties"]["battery_degradation_cost"]["default"] == 24.0
        assert schema["properties"]["round_trip_efficiency"]["default"] == 0.81
        assert schema["properties"]["timestep_in_hours"]["default"] == 1.0
        assert schema["properties"]["days"]["default"] == 365.0


    def test_tariffs_schema(self):
        from energbench.tools.tariffs_tool import TariffsTool

        tool = TariffsTool()
        td = tool.get_tools()[0]
        schema = td.parameters

        assert set(schema["required"]) == {"address", "sector"}

        sector = schema["properties"]["sector"]
        assert sector["type"] == "string"
        assert sector["enum"] == ["Residential", "Commercial", "Industrial", "Lighting"]

        fmt = schema["properties"]["return_format"]
        assert fmt["enum"] == ["json"]
        assert fmt["default"] == "json"

        ver = schema["properties"]["version"]
        assert ver["type"] == "integer"
        assert ver["enum"] == [7]
        assert ver["default"] == 7

        assert schema["properties"]["active_only"]["type"] == "boolean"
        assert schema["properties"]["active_only"]["default"] is True


    def test_search_web_schema(self):
        from energbench.tools.search_tool import SearchTool

        tool = SearchTool()
        td = next(t for t in tool.get_tools() if t.name == "search_web")
        schema = td.parameters

        assert schema["required"] == ["query"]
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["text"]["type"] == "boolean"
        assert schema["properties"]["text"]["default"] is True

        lc = schema["properties"]["livecrawl"]
        assert lc["type"] == "string"
        assert set(lc["enum"]) == {"never", "fallback", "preferred", "always"}

        st = schema["properties"]["search_type"]
        assert st["enum"] == ["neural", "fast", "auto", "deep"]

        assert schema["properties"]["include_domains"]["type"] == "array"

    def test_get_page_contents_schema(self):
        from energbench.tools.search_tool import SearchTool

        tool = SearchTool()
        td = next(t for t in tool.get_tools() if t.name == "get_page_contents")
        schema = td.parameters

        assert schema["required"] == ["urls"]
        assert schema["properties"]["urls"]["type"] == "array"

        text_prop = schema["properties"]["text"]
        assert "oneOf" in text_prop
        oneof_types = {s["type"] for s in text_prop["oneOf"]}
        assert oneof_types == {"boolean", "object"}


    def test_openweather_current_weather_schema(self):
        from energbench.tools.openweather_tool import OpenWeatherTool

        tool = OpenWeatherTool()
        td = next(t for t in tool.get_tools() if t.name == "get_current_weather")
        schema = td.parameters

        assert schema["required"] == ["location"]
        units = schema["properties"]["units"]
        assert units["type"] == "string"
        assert set(units["enum"]) == {"metric", "imperial", "standard"}


    def test_ferc_docket_schema(self):
        from energbench.tools.dockets.ferc_tool import FERCDocketTool

        tool = FERCDocketTool()
        td = tool.get_tools()[0]
        schema = td.parameters

        assert set(schema["required"]) == {"start_date", "end_date"}
        assert len(schema["properties"]) == 14

        at = schema["properties"]["affiliation_type"]
        assert at["type"] == "string"
        assert at["enum"] == ["agent", "author", "recipient"]

        assert schema["properties"]["sub_docket_numbers"]["type"] == "array"
        assert schema["properties"]["search_full_text"]["type"] == "boolean"
        assert schema["properties"]["search_full_text"]["default"] is True


    def test_system_tool_schemas(self):
        from energbench.tools.constants import SYSTEM_COMMAND_TIMEOUT, SYSTEM_MAX_RESULTS
        from energbench.tools.system_tool import SystemTool

        tool = SystemTool()
        tools_by_name = {t.name: t for t in tool.get_tools()}

        assert set(tools_by_name.keys()) == {
            "list_files", "grep_files", "run_python_code", "run_shell_command",
        }

        grep = tools_by_name["grep_files"].parameters
        assert grep["required"] == ["pattern"]
        assert grep["properties"]["max_results"]["default"] == SYSTEM_MAX_RESULTS

        shell = tools_by_name["run_shell_command"].parameters
        assert shell["required"] == ["command"]
        assert shell["properties"]["timeout"]["default"] == SYSTEM_COMMAND_TIMEOUT

        code = tools_by_name["run_python_code"].parameters
        assert code["required"] == ["code"]


    def test_renewables_solar_schema(self):
        from energbench.tools.renewables_tool import RenewablesTool

        tool = RenewablesTool()
        td = next(t for t in tool.get_tools() if t.name == "get_solar_profile")
        schema = td.parameters

        assert set(schema["required"]) == {"lat", "lon", "date_from", "date_to"}
        assert schema["properties"]["lat"]["type"] == "number"
        assert schema["properties"]["capacity"]["default"] == 1.0

        tracking = schema["properties"]["tracking"]
        assert tracking["type"] == "integer"
        assert tracking["enum"] == [0, 1, 2]
        assert tracking["default"] == 0

        fmt = schema["properties"]["format"]
        assert fmt["enum"] == ["json", "csv"]
