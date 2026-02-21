from dataclasses import dataclass, field


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server.
    """

    name: str
    command: str | None = None
    url: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate that either command or url is provided."""
        if not self.command and not self.url:
            raise ValueError(f"Server '{self.name}' must have either 'command' or 'url'")
