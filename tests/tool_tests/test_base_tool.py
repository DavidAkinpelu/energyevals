import pytest

from energbench.agent.schema import ToolDefinition
from energbench.tools.base_tool import BaseTool, ToolRegistry, ToolResult


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

            def get_tools(self):
                return []

        tool = TestTool()

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert len(tool._methods) == 0

    def test_register_method(self):
        """Test registering a method with a tool."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="test_tool", description="Test")
                self.register_method("test_method", self.test_method)

            def get_tools(self):
                return []

            def test_method(self, input: str) -> str:
                return f"Processed: {input}"

        tool = TestTool()

        assert "test_method" in tool._methods
        assert callable(tool._methods["test_method"])

    @pytest.mark.asyncio
    async def test_execute_sync_method(self):
        """Test executing a synchronous method."""

        class TestTool(BaseTool):
            def __init__(self):
                super().__init__(name="test_tool", description="Test")
                self.register_method("sync_method", self.sync_method)

            def get_tools(self):
                return []

            def sync_method(self, value: int) -> str:
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
                self.register_method("async_method", self.async_method)

            def get_tools(self):
                return []

            async def async_method(self, value: str) -> str:
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

            def get_tools(self):
                return []

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
                self.register_method("error_method", self.error_method)

            def get_tools(self):
                return []

            def error_method(self) -> str:
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

            def get_tools(self):
                return [
                    ToolDefinition(
                        name="test_method",
                        description="Test method",
                        parameters={"type": "object", "properties": {}},
                    )
                ]

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

            def get_tools(self):
                return [
                    ToolDefinition(name="method1", description="M1", parameters={})
                ]

        class Tool2(BaseTool):
            def __init__(self):
                super().__init__(name="tool2", description="Second")

            def get_tools(self):
                return [
                    ToolDefinition(name="method2", description="M2", parameters={})
                ]

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
                self.register_method("add", self.add)

            def get_tools(self):
                return [
                    ToolDefinition(
                        name="add",
                        description="Add two numbers",
                        parameters={
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["a", "b"],
                        },
                    )
                ]

            def add(self, a: int, b: int) -> int:
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

            def get_tools(self):
                return [
                    ToolDefinition(name="find_me", description="Target", parameters={}),
                    ToolDefinition(name="other", description="Other", parameters={}),
                ]

        registry = ToolRegistry()
        registry.register(TestTool())

        all_tools = registry.get_all_tools()
        target_tool = next((t for t in all_tools if t.name == "find_me"), None)

        assert target_tool is not None
        assert target_tool.description == "Target"

    def test_tool_registry_with_multiple_methods_per_tool(self):
        """Test a tool with multiple registered methods."""

        class MultiMethodTool(BaseTool):
            def __init__(self):
                super().__init__(name="multi", description="Multi-method tool")
                self.register_method("method1", self.method1)
                self.register_method("method2", self.method2)

            def get_tools(self):
                return [
                    ToolDefinition(name="method1", description="M1", parameters={}),
                    ToolDefinition(name="method2", description="M2", parameters={}),
                ]

            def method1(self) -> str:
                return "result1"

            def method2(self) -> str:
                return "result2"

        registry = ToolRegistry()
        registry.register(MultiMethodTool())

        all_tools = registry.get_all_tools()
        assert len(all_tools) == 2

        tool_names = {t.name for t in all_tools}
        assert "method1" in tool_names
        assert "method2" in tool_names
