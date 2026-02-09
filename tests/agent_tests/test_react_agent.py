from unittest.mock import AsyncMock, MagicMock

import pytest

from energbench.agent.react_agent import ReActAgent
from energbench.agent.schema import ToolDefinition
from energbench.tools.base_tool import BaseTool, ToolRegistry


class SimpleTestTool(BaseTool):
    """Simple test tool for ReActAgent tests."""

    def __init__(self):
        super().__init__(name="test_tool", description="A test tool")
        self.register_method("test_method", self.test_method)

    def get_tools(self):
        return [
            ToolDefinition(
                name="test_method",
                description="A test method",
                parameters={
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                    },
                    "required": ["input"],
                },
            )
        ]

    def test_method(self, input: str) -> str:
        return f"Result: {input}"


@pytest.fixture
def mock_provider():
    """Mock LLM provider for testing."""
    mock = MagicMock()
    mock.complete = AsyncMock()
    mock.model = "test-model"
    mock.provider_name = "mock"
    return mock


@pytest.fixture
def tool_registry():
    """Tool registry with a simple test tool."""
    registry = ToolRegistry()
    registry.register(SimpleTestTool())
    return registry


class TestReActAgentInitialization:
    """Tests for agent initialization."""

    def test_agent_init_with_defaults(self, mock_provider):
        """Test agent initialization with default parameters."""
        agent = ReActAgent(provider=mock_provider)

        assert agent.provider == mock_provider
        assert agent.max_iterations == 25
        assert agent.tools == []
        assert agent.tool_executor is not None

    def test_agent_init_with_custom_params(self, mock_provider, tool_registry):
        """Test agent initialization with custom parameters."""
        tools = tool_registry.get_all_tools()

        agent = ReActAgent(
            provider=mock_provider,
            tools=tools,
            max_iterations=20,
            csv_threshold=50,
        )

        assert agent.max_iterations == 20
        assert len(agent.tools) == len(tools)

    def test_agent_register_tool(self, mock_provider):
        """Test registering a tool after initialization."""
        agent = ReActAgent(provider=mock_provider)

        from energbench.agent.schema import ToolDefinition
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters={"type": "object", "properties": {}},
        )

        agent.register_tool(tool)
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "test_tool"

    def test_agent_register_multiple_tools(self, mock_provider, tool_registry):
        """Test registering multiple tools."""
        agent = ReActAgent(provider=mock_provider)
        tools = tool_registry.get_all_tools()

        agent.register_tools(tools)
        assert len(agent.tools) == len(tools)


class TestAgentMessageBuilding:
    """Tests for message building functionality."""

    def test_build_initial_messages_no_context(self, mock_provider):
        """Test building initial messages without context."""
        agent = ReActAgent(provider=mock_provider)

        messages = agent._build_initial_messages(query="Test query")

        assert len(messages) == 2  # System + user
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "Test query" in messages[1].content

    def test_build_initial_messages_with_context(self, mock_provider):
        """Test building initial messages with context."""
        agent = ReActAgent(provider=mock_provider)

        messages = agent._build_initial_messages(
            query="Test query",
            context={"info": "Previous context information"}
        )

        assert len(messages) == 2
        assert "Previous context information" in messages[1].content
        assert "Test query" in messages[1].content


class TestToolExecution:
    """Tests for tool execution functionality."""

    def test_default_tool_executor_unknown_tool(self, mock_provider):
        """Test default tool executor with unknown tool."""
        agent = ReActAgent(provider=mock_provider)

        result = agent._default_tool_executor("unknown_tool", {})

        assert "error" in result.lower()
        assert "no tool executor configured" in result.lower()

    def test_extract_images_from_result_no_images(self, mock_provider):
        """Test image extraction with result containing no images."""
        agent = ReActAgent(provider=mock_provider)

        result = '{"data": "some text", "count": 5}'
        images = agent._extract_images_from_result(result)

        assert isinstance(images, list)
        assert len(images) == 0

    def test_extract_images_from_result_invalid_json(self, mock_provider):
        """Test image extraction with invalid JSON."""
        agent = ReActAgent(provider=mock_provider)

        result = "Not a JSON string"
        images = agent._extract_images_from_result(result)

        assert isinstance(images, list)
        assert len(images) == 0
