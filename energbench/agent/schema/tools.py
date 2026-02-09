from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from energbench.agent.constants import CSV_THRESHOLD, PREVIEW_ROWS


@dataclass
class ToolCall:
    """Represents a tool call from the model.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool being called.
        arguments: Arguments passed to the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolDefinition:
    """Definition of a tool for the LLM.

    Attributes:
        name: Unique name of the tool.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema defining the tool's parameters.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


ToolExecutor = Callable[[str, dict[str, Any]], Union[str, Awaitable[str]]]


@dataclass
class ToolResult:
    """Result from a tool execution.

    Attributes:
        success: Whether the tool executed successfully.
        data: The result data (can be dict, list, or string).
        error: Error message if execution failed.
        row_count: Number of rows if result is tabular data.
        csv_path: Path to CSV file if data was saved.
        metadata: Additional metadata about the result.
    """

    success: bool
    data: Any
    error: Optional[str] = None
    row_count: int = 0
    csv_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_string(self, csv_threshold: int = CSV_THRESHOLD) -> str:
        """Convert result to a string suitable for LLM context.

        If the result has more rows than csv_threshold and was saved to CSV,
        returns a reference to the CSV file instead of the full data.

        Args:
            csv_threshold: Row count threshold for using CSV reference.

        Returns:
            String representation of the result for LLM context.
        """
        import json

        if not self.success:
            return json.dumps({"error": self.error})

        if self.csv_path and self.row_count > csv_threshold:
            return json.dumps({
                "status": "success",
                "row_count": self.row_count,
                "csv_file": self.csv_path,
                "message": f"Results saved to {self.csv_path}. Use Python to read and analyze the CSV file.",
                "preview": self.data[:PREVIEW_ROWS] if isinstance(self.data, list) else None,
            }, indent=2, default=str)

        return json.dumps(self.data, indent=2, default=str)
