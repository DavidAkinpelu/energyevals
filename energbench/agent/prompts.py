"""System prompts for the ReAct agent."""

SYSTEM_PROMPT = """You are a Senior Energy Analyst. 
"""


def get_system_prompt(
    role: str = "default",
    custom_instructions: str | None = None,
) -> str:
    """Get the appropriate system prompt for a given role.

    Args:
        role: The role/specialization ("default", "analyst", "battery", "regulatory").
        custom_instructions: Optional additional instructions to append.

    Returns:
        The formatted system prompt.
    """
  
    prompt = SYSTEM_PROMPT

    if custom_instructions:
        prompt = f"{prompt}\n\n## Additional Instructions\n{custom_instructions}"

    return prompt
