import pytest

from energbench.agent.providers import Message, ToolDefinition
from energbench.agent.schema.messages import ImageContent, TextContent


@pytest.fixture
def sample_tool():
    """Sample tool definition for testing."""
    return ToolDefinition(
        name="get_energy_price",
        description="Get real-time energy price",
        parameters={
            "type": "object",
            "properties": {
                "market": {"type": "string", "enum": ["ERCOT", "PJM", "CAISO"]},
                "zone": {"type": "string"},
            },
            "required": ["market"],
        },
    )


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message(role="system", content="You are an energy analyst."),
        Message(role="user", content="What is the current energy price in ERCOT?"),
    ]


@pytest.fixture
def sample_image_message():
    """Sample message with a tiny 1x1 red PNG for multimodal testing."""
    # 1x1 red PNG (minimal valid PNG, 67 bytes)
    tiny_png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    return Message(
        role="user",
        content_parts=[
            TextContent(text="Describe this image briefly"),
            ImageContent(image_base64=tiny_png_base64, media_type="image/png"),
        ],
    )
