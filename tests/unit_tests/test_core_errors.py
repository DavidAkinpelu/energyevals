from energbench.core.errors import (
    APIError,
    ConfigurationError,
    EnergBenchError,
    ProviderError,
    ToolError,
)


def test_energbench_error_basic():
    """Test basic EnergBenchError creation."""
    error = EnergBenchError("Test error")
    assert str(error) == "Test error"
    assert error.message == "Test error"
    assert error.context == {}


def test_energbench_error_with_context():
    """Test EnergBenchError with context."""
    error = EnergBenchError("Test error", context={"key": "value"})
    assert "Test error" in str(error)
    assert "key=value" in str(error)
    assert error.context == {"key": "value"}


def test_tool_error():
    """Test ToolError creation."""
    error = ToolError("Test error", tool_name="test_tool", recoverable=True)
    assert error.tool_name == "test_tool"
    assert error.recoverable is True
    assert error.context["tool_name"] == "test_tool"
    assert error.context["recoverable"] is True


def test_api_error():
    """Test APIError creation."""
    error = APIError(
        "API request failed",
        tool_name="test_tool",
        status_code=404,
        response_body="Not Found",
    )
    assert error.status_code == 404
    assert error.response_body == "Not Found"
    assert error.tool_name == "test_tool"
    assert error.recoverable is True
    assert error.context["status_code"] == 404


def test_provider_error():
    """Test ProviderError creation."""
    error = ProviderError(
        "Provider failed",
        provider="openai",
        model="gpt-4",
        recoverable=False,
    )
    assert error.provider == "openai"
    assert error.model == "gpt-4"
    assert error.recoverable is False
    assert error.context["provider"] == "openai"
    assert error.context["model"] == "gpt-4"


def test_configuration_error():
    """Test ConfigurationError creation."""
    error = ConfigurationError("Missing config", config_key="api_key")
    assert error.config_key == "api_key"
    assert error.context["config_key"] == "api_key"


def test_error_inheritance():
    """Test error inheritance hierarchy."""
    assert issubclass(ToolError, EnergBenchError)
    assert issubclass(APIError, ToolError)
    assert issubclass(ProviderError, EnergBenchError)
    assert issubclass(ConfigurationError, EnergBenchError)
