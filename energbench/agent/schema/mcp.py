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
        """Validate remote-only MCP server configuration."""
        if not self.url:
            raise ValueError(f"Server '{self.name}' must provide a remote 'url'")
        if self.command:
            raise ValueError(
                f"Server '{self.name}' is configured with 'command', "
                "but this MCP client supports remote URL servers only"
            )
