from typing import Any

from energbench.agent.schema import ImageContent, Message, TextContent


def separate_system_message(
    messages: list[Message],
) -> tuple[str, list[Message]]:
    """Separate system message from conversation messages.

    Many providers (Anthropic, Google) require system messages to be
    passed separately from conversation messages.

    Args:
        messages: List of messages including potential system message

    Returns:
        Tuple of (system_message_text, remaining_conversation_messages)
    """
    system_msg = ""
    conversation = []

    for msg in messages:
        if msg.role == "system":
            system_msg = msg.content
        else:
            conversation.append(msg)

    return system_msg, conversation


def format_multimodal_content(
    content_parts: list[TextContent | ImageContent],
) -> list[dict[str, Any]]:
    """Format multimodal content (text + images) for API requests.

    Handles conversion of TextContent and ImageContent objects into
    the format expected by provider APIs.

    Args:
        content_parts: List of content parts (text or image)

    Returns:
        List of formatted content blocks for API
    """
    formatted = []

    for part in content_parts:
        if isinstance(part, TextContent):
            formatted.append({
                "type": "text",
                "text": part.text,
            })
        elif isinstance(part, ImageContent):
            image_block: dict[str, Any] = {
                "type": "image",
            }

            if part.image_url:
                image_block["source"] = {
                    "type": "url",
                    "url": part.image_url,
                }
            elif part.image_base64:
                media_type = part.media_type or "image/jpeg"
                image_block["source"] = {
                    "type": "base64",
                    "media_type": media_type,
                    "data": part.image_base64,
                }

            formatted.append(image_block)

    return formatted


def format_tool_calls_for_openai(
    tool_calls: list[Any],
) -> list[dict[str, Any]]:
    """Format tool calls for OpenAI-compatible APIs.

    Args:
        tool_calls: List of tool call objects

    Returns:
        List of formatted tool calls
    """
    formatted = []

    for tool_call in tool_calls:
        formatted.append({
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
        })

    return formatted
