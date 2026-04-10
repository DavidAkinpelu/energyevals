import pytest

from energyevals.agent.exceptions import (
    AgentError,
    ConfigurationError,
    ProviderError,
    ToolExecutionError,
)


class TestAgentError:
    """Tests for base AgentError class."""

    def test_basic_agent_error(self):
        """Test creating basic AgentError."""
        error = AgentError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"

    def test_agent_error_with_details(self):
        """Test AgentError with additional context."""
        error = AgentError("Test error", context={"extra_info": "details"})
        assert error.message == "Test error"
        assert error.context["extra_info"] == "details"


class TestProviderError:
    """Tests for ProviderError class."""

    def test_provider_error_basic(self):
        """Test creating ProviderError."""
        error = ProviderError("API failed", provider="openai", model="gpt-4")

        assert error.message == "API failed"
        assert error.provider == "openai"
        assert error.model == "gpt-4"
        assert "API failed" in str(error)

    def test_provider_error_without_model(self):
        """Test ProviderError without model specified."""
        error = ProviderError("Connection error", provider="anthropic")

        assert error.message == "Connection error"
        assert error.provider == "anthropic"
        assert error.model is None

    def test_provider_error_inheritance(self):
        """Test that ProviderError inherits from AgentError."""
        error = ProviderError("Test", provider="test")
        assert isinstance(error, AgentError)


class TestToolExecutionError:
    """Tests for ToolExecutionError class."""

    def test_tool_execution_error_basic(self):
        """Test creating ToolExecutionError."""
        error = ToolExecutionError(
            "Tool failed",
            tool_name="search_web",
            arguments={"query": "test"},
        )

        assert error.message == "Tool failed"
        assert error.tool_name == "search_web"
        assert error.arguments == {"query": "test"}
        assert "Tool failed" in str(error)

    def test_tool_execution_error_without_arguments(self):
        """Test ToolExecutionError without arguments."""
        error = ToolExecutionError("Tool failed", tool_name="test_tool")

        assert error.message == "Tool failed"
        assert error.tool_name == "test_tool"
        assert error.arguments is None

    def test_tool_execution_error_with_complex_arguments(self):
        """Test ToolExecutionError with nested arguments."""
        args = {
            "query": "test",
            "filters": {"date": "2024-01-01", "type": "article"},
            "limit": 10,
        }
        error = ToolExecutionError("Parse error", tool_name="search", arguments=args)

        assert error.arguments == args
        assert error.arguments["filters"]["date"] == "2024-01-01"

    def test_tool_execution_error_inheritance(self):
        """Test that ToolExecutionError inherits from AgentError."""
        error = ToolExecutionError("Test", tool_name="test")
        assert isinstance(error, AgentError)


class TestConfigurationError:
    """Tests for ConfigurationError class."""

    def test_configuration_error_basic(self):
        """Test creating ConfigurationError."""
        error = ConfigurationError("API key missing", config_key="OPENAI_API_KEY")

        assert error.message == "API key missing"
        assert error.config_key == "OPENAI_API_KEY"
        assert "API key missing" in str(error)

    def test_configuration_error_without_key(self):
        """Test ConfigurationError without config_key."""
        error = ConfigurationError("Invalid configuration")

        assert error.message == "Invalid configuration"
        assert error.config_key is None

    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from AgentError."""
        error = ConfigurationError("Test")
        assert isinstance(error, AgentError)


class TestExceptionUsage:
    """Tests for practical exception usage patterns."""

    def test_raising_and_catching_provider_error(self):
        """Test raising and catching ProviderError."""
        with pytest.raises(ProviderError) as exc_info:
            raise ProviderError("Rate limit", provider="openai", model="gpt-4")

        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "gpt-4"

    def test_catching_as_agent_error(self):
        """Test catching specific errors as base AgentError."""
        with pytest.raises(AgentError):
            raise ToolExecutionError("Failed", tool_name="test")

    def test_exception_in_try_except(self):
        """Test exception handling in try/except blocks."""
        try:
            raise ToolExecutionError(
                "Tool timeout",
                tool_name="slow_tool",
                arguments={"param": "value"},
            )
        except ToolExecutionError as e:
            assert e.tool_name == "slow_tool"
            assert e.arguments["param"] == "value"
        except AgentError:
            pytest.fail("Should catch specific exception type first")
