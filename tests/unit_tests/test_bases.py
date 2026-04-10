import pytest

from energyevals.agent.providers import ToolDefinition
from energyevals.core.errors import ConfigurationError
from energyevals.tools.bases import APITool


class ConcreteAPITool(APITool):
    """Concrete subclass of APITool for testing."""

    def get_tools(self):
        return [
            ToolDefinition(
                name="test_method",
                description="A test method",
                parameters={"type": "object", "properties": {}},
            )
        ]


class TestAPITool:
    """Tests for APITool base class."""

    def test_api_tool_creation_without_key(self, monkeypatch):
        """Test creating APITool without API key set."""
        monkeypatch.delenv("TEST_API_KEY", raising=False)

        tool = ConcreteAPITool(
            name="test",
            description="Test tool",
            api_key_env_var="TEST_API_KEY",
        )

        assert tool.api_key is None
        assert tool.name == "test"

    def test_api_tool_creation_with_key(self, monkeypatch):
        """Test creating APITool with API key set."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-123")

        tool = ConcreteAPITool(
            name="test",
            description="Test tool",
            api_key_env_var="TEST_API_KEY",
        )

        assert tool.api_key == "test-key-123"

    def test_require_api_key_raises_when_missing(self, monkeypatch):
        """Test that require_api_key raises when key is missing."""
        monkeypatch.delenv("TEST_API_KEY", raising=False)

        tool = ConcreteAPITool(
            name="test",
            description="Test tool",
            api_key_env_var="TEST_API_KEY",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            tool.require_api_key()

        assert "TEST_API_KEY" in str(exc_info.value)

    def test_require_api_key_returns_key_when_set(self, monkeypatch):
        """Test that require_api_key returns key when set."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-123")

        tool = ConcreteAPITool(
            name="test",
            description="Test tool",
            api_key_env_var="TEST_API_KEY",
        )

        key = tool.require_api_key()
        assert key == "test-key-123"
