from energbench.agent.providers.provider_utils import (
    format_multimodal_content,
    format_tool_calls_for_openai,
    separate_system_message,
)
from energbench.agent.schema import ImageContent, Message, TextContent, ToolCall


class TestSeparateSystemMessage:
    """Tests for separate_system_message function."""

    def test_single_system_message(self):
        """Test separating a single system message."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
        ]

        system_text, conversation = separate_system_message(messages)

        assert system_text == "You are a helpful assistant."
        assert len(conversation) == 1
        assert conversation[0].role == "user"

    def test_multiple_system_messages(self):
        """Test that multiple system messages keep the last one."""
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="system", content="You are knowledgeable."),
            Message(role="user", content="Question"),
        ]

        system_text, conversation = separate_system_message(messages)

        assert system_text == "You are knowledgeable."
        assert len(conversation) == 1

    def test_no_system_message(self):
        """Test when there's no system message."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]

        system_text, conversation = separate_system_message(messages)

        assert system_text == ""
        assert len(conversation) == 2

    def test_system_message_in_middle(self):
        """Test system message not at the beginning."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="system", content="Be concise."),
            Message(role="assistant", content="Hi"),
        ]

        system_text, conversation = separate_system_message(messages)

        assert system_text == "Be concise."
        assert len(conversation) == 2

    def test_empty_messages(self):
        """Test with empty message list."""
        messages = []

        system_text, conversation = separate_system_message(messages)

        assert system_text == ""
        assert len(conversation) == 0


class TestFormatMultimodalContent:
    """Tests for format_multimodal_content function."""

    def test_text_only_content(self):
        """Test formatting text-only content."""
        content = [TextContent(type="text", text="Hello")]

        formatted = format_multimodal_content(content)

        assert isinstance(formatted, list)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "text"
        assert formatted[0]["text"] == "Hello"

    def test_image_url_content(self):
        """Test formatting image URL content."""
        content = [
            TextContent(type="text", text="Check this image:"),
            ImageContent(type="image", image_url="https://example.com/img.jpg"),
        ]

        formatted = format_multimodal_content(content)

        assert len(formatted) == 2
        assert formatted[0]["type"] == "text"
        assert formatted[1]["type"] == "image"
        assert formatted[1]["source"]["type"] == "url"
        assert formatted[1]["source"]["url"] == "https://example.com/img.jpg"

    def test_base64_image_content(self):
        """Test formatting base64 image content."""
        content = [
            ImageContent(
                type="image",
                image_base64="iVBORw0KGgo...",
                media_type="image/png",
            ),
        ]

        formatted = format_multimodal_content(content)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "image"
        assert formatted[0]["source"]["type"] == "base64"
        assert formatted[0]["source"]["data"] == "iVBORw0KGgo..."
        assert formatted[0]["source"]["media_type"] == "image/png"

    def test_mixed_content(self):
        """Test formatting mixed text and images."""
        content = [
            TextContent(type="text", text="First text"),
            ImageContent(type="image", image_url="https://example.com/img1.jpg"),
            TextContent(type="text", text="Second text"),
            ImageContent(type="image", image_url="https://example.com/img2.jpg"),
        ]

        formatted = format_multimodal_content(content)

        assert len(formatted) == 4
        assert formatted[0]["type"] == "text"
        assert formatted[1]["type"] == "image"
        assert formatted[2]["type"] == "text"
        assert formatted[3]["type"] == "image"

    def test_empty_content(self):
        """Test formatting empty content list."""
        content = []

        formatted = format_multimodal_content(content)

        assert formatted == []


class TestFormatToolCallsForOpenAI:
    """Tests for format_tool_calls_for_openai function."""

    def test_single_tool_call(self):
        """Test formatting a single tool call."""

        tool_calls = [
            ToolCall(
                id="call_123",
                name="get_weather",
                arguments={"location": "Austin"},
            )
        ]

        formatted = format_tool_calls_for_openai(tool_calls)

        assert len(formatted) == 1
        assert formatted[0]["id"] == "call_123"
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "get_weather"
        assert formatted[0]["function"]["arguments"] == {"location": "Austin"}

    def test_multiple_tool_calls(self):
        """Test formatting multiple tool calls."""

        tool_calls = [
            ToolCall(id="call_1", name="tool1", arguments={"a": 1}),
            ToolCall(id="call_2", name="tool2", arguments={"b": 2}),
        ]

        formatted = format_tool_calls_for_openai(tool_calls)

        assert len(formatted) == 2
        assert formatted[0]["id"] == "call_1"
        assert formatted[1]["id"] == "call_2"

    def test_empty_arguments(self):
        """Test formatting tool call with empty arguments."""

        tool_calls = [
            ToolCall(id="call_1", name="no_args_tool", arguments={})
        ]

        formatted = format_tool_calls_for_openai(tool_calls)

        assert formatted[0]["function"]["arguments"] == {}

    def test_complex_arguments(self):
        """Test formatting tool call with nested arguments."""

        tool_calls = [
            ToolCall(
                id="call_1",
                name="complex_tool",
                arguments={
                    "query": "test",
                    "filters": {"date": "2024-01-01", "type": "news"},
                    "limit": 10,
                },
            )
        ]

        formatted = format_tool_calls_for_openai(tool_calls)

        args = formatted[0]["function"]["arguments"]
        assert args["query"] == "test"
        assert args["filters"]["date"] == "2024-01-01"
        assert args["limit"] == 10
