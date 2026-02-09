from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server.
    """

    name: str
    command: Optional[str] = None
    url: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: Optional[dict[str, str]] = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate that either command or url is provided."""
        if not self.command and not self.url:
            raise ValueError(f"Server '{self.name}' must have either 'command' or 'url'")
