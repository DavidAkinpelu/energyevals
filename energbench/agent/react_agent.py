"""Custom ReAct agent implementation."""

import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from .prompts import get_system_prompt
from .providers import BaseProvider, ProviderResponse, ToolDefinition
from .schema import (
    AgentConfig,
    AgentRun,
    AgentStep,
    ImageContent,
    Message,
    StepType,
    TextContent,
    ToolExecutor,
    ToolResult,
)


class ReActAgent:
    """Custom ReAct agent with multi-provider support.

    This agent implements the ReAct (Reasoning and Acting) pattern,
    alternating between thinking about the problem and taking actions
    via tools to gather information.
    """

    def __init__(
        self,
        provider: BaseProvider,
        tools: Optional[list[ToolDefinition]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        max_iterations: int = 50,
        system_prompt: Optional[str] = None,
        csv_threshold: int = 20,
        csv_output_dir: str = "./agent_outputs",
    ):
        """Initialize the ReAct agent.

        Args:
            provider: The LLM provider to use.
            tools: List of available tools.
            tool_executor: Function to execute tool calls.
            max_iterations: Maximum number of iterations before stopping.
            system_prompt: Custom system prompt. If None, uses default prompt.
            csv_threshold: Row count threshold for saving results to CSV (default: 20).
            csv_output_dir: Directory to save CSV files (default: "./agent_outputs").
        """
        self.provider = provider
        self.tools = tools or []
        self.tool_executor = tool_executor or self._default_tool_executor
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or get_system_prompt()
        self.csv_threshold = csv_threshold
        self.csv_output_dir = csv_output_dir

        # Ensure output directory exists
        os.makedirs(self.csv_output_dir, exist_ok=True)

        # Tool registry for quick lookup
        self._tool_registry: dict[str, ToolDefinition] = {t.name: t for t in self.tools}

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a new tool with the agent.

        Args:
            tool: The tool definition to register.
        """
        self.tools.append(tool)
        self._tool_registry[tool.name] = tool

    def register_tools(self, tools: list[ToolDefinition]) -> None:
        """Register multiple tools with the agent.

        Args:
            tools: List of tool definitions to register.
        """
        for tool in tools:
            self.register_tool(tool)

    async def run(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
    ) -> AgentRun:
        """Execute the ReAct loop for a given query.

        Args:
            query: The user's query to process.
            context: Optional additional context to include.

        Returns:
            AgentRun containing the full execution trace and result.
        """
        run = AgentRun(query=query)

        # Build initial messages
        messages = self._build_initial_messages(query, context)

        logger.info(f"Starting agent run for query: {query[:100]}...")

        try:
            for iteration in range(self.max_iterations):
                run.iterations = iteration + 1

                # Get model response
                response = await self._get_response(messages)

                # Update metrics
                run.total_input_tokens += response.input_tokens
                run.total_cached_tokens += response.cached_tokens
                run.total_output_tokens += response.output_tokens
                run.total_latency_ms += response.latency_ms

                # Process response
                if response.tool_calls:
                    # Handle tool calls
                    await self._process_tool_calls(
                        response, messages, run
                    )
                else:
                    # No tool calls - this is the final answer
                    run.final_answer = response.content
                    run.steps.append(
                        AgentStep(
                            step_type=StepType.ANSWER,
                            content=response.content,
                            tokens_used=response.input_tokens + response.output_tokens,
                            latency_ms=response.latency_ms,
                        )
                    )
                    run.success = True
                    break

            if not run.success and run.iterations >= self.max_iterations:
                run.error = f"Max iterations ({self.max_iterations}) reached"
                logger.warning(run.error)

        except Exception as e:
            run.error = str(e)
            run.steps.append(
                AgentStep(
                    step_type=StepType.ERROR,
                    content=str(e),
                )
            )
            logger.error(f"Agent run failed: {e}")

        run.end_time = time.time()
        logger.info(
            f"Agent run completed: success={run.success}, "
            f"iterations={run.iterations}, tokens={run.total_tokens}"
        )

        return run

    def _build_initial_messages(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
    ) -> list[Message]:
        """Build the initial message list for the conversation."""
        messages = [
            Message(role="system", content=self._build_system_prompt()),
        ]

        # Add context if provided
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            messages.append(
                Message(
                    role="user",
                    content=f"Context:\n{context_str}\n\nQuery: {query}",
                )
            )
        else:
            messages.append(Message(role="user", content=query))

        return messages

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions."""
        if not self.tools:
            return self.system_prompt

        tool_descriptions = "\n".join(
            f"- **{tool.name}**: {tool.description}" for tool in self.tools
        )

        print(f"{self.system_prompt}\n\n## Available Tools\n{tool_descriptions}")

        return f"{self.system_prompt}\n\n## Available Tools\n{tool_descriptions}"

    async def _get_response(self, messages: list[Message]) -> ProviderResponse:
        """Get a response from the provider."""
        return await self.provider.complete(
            messages=messages,
            tools=self.tools if self.tools else None,
            temperature=0.0,
        )

    async def _process_tool_calls(
        self,
        response: ProviderResponse,
        messages: list[Message],
        run: AgentRun,
    ) -> None:
        """Process tool calls from the model response."""
        # Add assistant message with tool calls
        messages.append(
            Message(
                role="assistant",
                content=response.content,
                tool_calls=[
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in response.tool_calls
                ]
                if response.tool_calls
                else None,
            )
        )

        # Execute each tool call
        for tool_call in response.tool_calls or []:
            run.tool_calls_count += 1

            # Record action step
            action_step = AgentStep(
                step_type=StepType.ACTION,
                content=f"Calling {tool_call.name}",
                tool_name=tool_call.name,
                tool_input=tool_call.arguments,
                latency_ms=response.latency_ms,
            )
            run.steps.append(action_step)

            logger.debug(f"Executing tool: {tool_call.name}")
            logger.debug(f"Tool arguments: {json.dumps(tool_call.arguments, indent=2)}")

            # Execute the tool
            start_time = time.time()
            try:
                tool_result = await self._execute_tool(
                    tool_call.name, tool_call.arguments
                )
            except Exception as e:
                tool_result = json.dumps({"error": str(e)})
                logger.error(f"Tool execution failed: {e}")

            execution_time = (time.time() - start_time) * 1000

            # Process result - save to CSV if needed
            context_result, csv_path = self._process_tool_result(
                tool_call.name, tool_result
            )

            # Log truncated result for console readability
            log_preview = tool_result[:500] + "..." if len(tool_result) > 500 else tool_result
            logger.debug(f"Tool {tool_call.name} result (truncated): {log_preview}")
            if csv_path:
                logger.info(f"Large result saved to CSV: {csv_path}")

            # Record observation step with full output (for observability)
            obs_step = AgentStep(
                step_type=StepType.OBSERVATION,
                content=context_result,
                tool_name=tool_call.name,
                tool_output=tool_result,  # Full result stored for Langfuse
                latency_ms=execution_time,
            )
            run.steps.append(obs_step)

            # Add processed result to messages (may include images)
            tool_message = self._create_tool_message(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=tool_result,
                context_result=context_result,
            )
            messages.append(tool_message)

    def _create_tool_message(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
        context_result: str,
    ) -> Message:
        """Create a tool message, handling images from RAG results.

        Args:
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Full raw result from the tool.
            context_result: Processed result for context (may exclude large data).

        Returns:
            Message with text and optional images.
        """
        # Check if result contains images (from RAG server)
        images = self._extract_images_from_result(result)

        if not images:
            return Message(
                role="tool",
                content=context_result,
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        # Multi-modal message with text and images
        content_parts = [TextContent(text=context_result)]

        for img in images:
            content_parts.append(
                ImageContent(
                    image_base64=img["base64"],
                    media_type=img.get("media_type", "image/jpeg"),
                )
            )

        logger.info(f"Tool {tool_name} returned {len(images)} image(s)")

        return Message(
            role="tool",
            content=context_result,  # Keep text content for providers that don't support images
            content_parts=content_parts,
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    def _extract_images_from_result(self, result: str) -> list[dict[str, str]]:
        """Extract base64 images from a tool result.

        Looks for image_base64 fields in RAG search results.

        Args:
            result: JSON result string from tool.

        Returns:
            List of dicts with 'base64' and 'media_type' keys.
        """
        images = []

        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return images

        # Handle RAG search results format
        if isinstance(data, dict):
            documents = data.get("documents", [])
            for doc in documents:
                if isinstance(doc, dict) and "image_base64" in doc:
                    images.append({
                        "base64": doc["image_base64"],
                        "media_type": doc.get("media_type", "image/jpeg"),
                    })

        return images

    def _process_tool_result(
        self,
        tool_name: str,
        result: str,
    ) -> tuple[str, Optional[str]]:
        """Process tool result, saving to CSV if it exceeds threshold.

        Args:
            tool_name: Name of the tool that produced the result.
            result: Raw JSON result string from the tool.

        Returns:
            Tuple of (context_result, csv_path). context_result is what gets
            sent to the LLM context. csv_path is the path to CSV if saved.
        """
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return result, None

        # Check if this is a database query result with rows
        rows = None
        columns = None

        if isinstance(data, dict):
            rows = data.get("rows")
            columns = data.get("columns")

        if not rows or not isinstance(rows, list):
            return result, None

        row_count = len(rows)

        # If under threshold, return as-is
        if row_count <= self.csv_threshold:
            return result, None

        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{tool_name}_{timestamp}.csv"
        csv_path = os.path.join(self.csv_output_dir, csv_filename)

        try:
            with open(csv_path, "w", newline="") as f:
                if rows and isinstance(rows[0], dict):
                    # Rows are dictionaries
                    fieldnames = list(rows[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                elif columns:
                    # Rows are lists with separate columns
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(rows)
                else:
                    return result, None

            # Create summary for context
            preview_rows = rows[:5]
            context_data = {
                "status": "success",
                "row_count": row_count,
                "csv_file": csv_path,
                "message": f"Query returned {row_count} rows. Results saved to {csv_path}. Use Python to read and analyze the CSV file.",
                "columns": columns or (list(rows[0].keys()) if rows else []),
                "preview": preview_rows,
            }

            # Include other metadata from original result
            for key in ["database", "query", "table"]:
                if key in data:
                    context_data[key] = data[key]

            return json.dumps(context_data, indent=2, default=str), csv_path

        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return result, None

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Execute a tool and return its result."""
        if tool_name not in self._tool_registry:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        result = self.tool_executor(tool_name, arguments)

        # Handle async results
        if hasattr(result, "__await__"):
            result = await result

        # Ensure string result
        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)
        return str(result)

    def _default_tool_executor(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Default tool executor that returns an error.

        Override this by providing a custom tool_executor to the constructor.
        """
        return json.dumps(
            {
                "error": "No tool executor configured",
                "tool": tool_name,
                "arguments": arguments,
            }
        )


class AgentBuilder:
    """Builder pattern for creating ReAct agents."""

    def __init__(self) -> None:
        self._provider: Optional[BaseProvider] = None
        self._tools: list[ToolDefinition] = []
        self._tool_executor: Optional[ToolExecutor] = None
        self._max_iterations: int = 10
        self._system_prompt: Optional[str] = None
        self._csv_threshold: int = 20
        self._csv_output_dir: str = "./agent_outputs"

    def with_provider(self, provider: BaseProvider) -> "AgentBuilder":
        """Set the LLM provider."""
        self._provider = provider
        return self

    def with_tool(self, tool: ToolDefinition) -> "AgentBuilder":
        """Add a tool to the agent."""
        self._tools.append(tool)
        return self

    def with_tools(self, tools: list[ToolDefinition]) -> "AgentBuilder":
        """Add multiple tools to the agent."""
        self._tools.extend(tools)
        return self

    def with_tool_executor(self, executor: ToolExecutor) -> "AgentBuilder":
        """Set the tool executor function."""
        self._tool_executor = executor
        return self

    def with_max_iterations(self, max_iterations: int) -> "AgentBuilder":
        """Set the maximum number of iterations."""
        self._max_iterations = max_iterations
        return self

    def with_system_prompt(self, prompt: str) -> "AgentBuilder":
        """Set a custom system prompt."""
        self._system_prompt = prompt
        return self

    def with_csv_threshold(self, threshold: int) -> "AgentBuilder":
        """Set the row count threshold for saving results to CSV."""
        self._csv_threshold = threshold
        return self

    def with_csv_output_dir(self, directory: str) -> "AgentBuilder":
        """Set the directory for CSV output files."""
        self._csv_output_dir = directory
        return self

    def with_config(self, config: AgentConfig) -> "AgentBuilder":
        """Apply configuration from an AgentConfig object."""
        self._max_iterations = config.max_iterations
        self._csv_threshold = config.csv_threshold
        self._csv_output_dir = config.csv_output_dir
        self._system_prompt = config.system_prompt
        return self

    def build(self) -> ReActAgent:
        """Build the configured agent."""
        if self._provider is None:
            raise ValueError("Provider is required")

        return ReActAgent(
            provider=self._provider,
            tools=self._tools,
            tool_executor=self._tool_executor,
            max_iterations=self._max_iterations,
            system_prompt=self._system_prompt,
            csv_threshold=self._csv_threshold,
            csv_output_dir=self._csv_output_dir,
        )
