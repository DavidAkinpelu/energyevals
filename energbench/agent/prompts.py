SYSTEM_PROMPT = """You are an Expert Energy Analyst.
Use your best effort to answer each question with only one attempt.
No room for back and forths with the user.
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
