from __future__ import annotations

SYSTEM_PROMPT = """You are a Senior Energy Analyst.
"""


def get_system_prompt(
    custom_instructions: str | None = None,
) -> str:
    """Get the system prompt with optional custom instructions.

    Args:
        custom_instructions: Optional additional instructions to append.

    Returns:
        The formatted system prompt.
    """
    prompt = SYSTEM_PROMPT

    if custom_instructions:
        prompt = f"{prompt}\n\n## Additional Instructions\n{custom_instructions}"

    return prompt
