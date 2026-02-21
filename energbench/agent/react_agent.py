import json
import time
from typing import Any, Self

from loguru import logger

from .constants import (
    CSV_THRESHOLD,
    MAX_ITERATIONS,
    QUERY_TRUNCATE_LENGTH,
    TOOL_RESULT_PREVIEW_LENGTH,
)
from .processors import ResultProcessor
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
        tools: list[ToolDefinition] | None = None,
        tool_executor: ToolExecutor | None = None,
        max_iterations: int = MAX_ITERATIONS,
        system_prompt: str | None = None,
        csv_threshold: int = CSV_THRESHOLD,
        csv_output_dir: str = "./agent_outputs",
        result_processor: ResultProcessor | None = None,
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
            result_processor: Custom result processor. If None, creates default.
        """
        self.provider = provider
        self.tools = tools or []
        self.tool_executor = tool_executor or self._default_tool_executor
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or get_system_prompt()
        self._result_processor = result_processor or ResultProcessor(
            csv_threshold=csv_threshold,
            csv_output_dir=csv_output_dir,
        )
        self._tool_registry: dict[str, ToolDefinition] = {t.name: t for t in self.tools}

    def register_tool(self, tool: ToolDefinition) -> None:
        self.tools.append(tool)
        self._tool_registry[tool.name] = tool

    def register_tools(self, tools: list[ToolDefinition]) -> None:
        for tool in tools:
            self.register_tool(tool)

    async def run(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> AgentRun:
        """Execute the ReAct loop for a given query.

        Args:
            query: The user's query to process.
            context: Optional additional context to include.

        Returns:
            AgentRun containing the full execution trace and result.
        """
        run = AgentRun(query=query)
        messages = self._build_initial_messages(query, context)

        logger.debug(f"Starting agent run for query: {query[:QUERY_TRUNCATE_LENGTH]}...")

        try:
            for iteration in range(self.max_iterations):
                run.iterations = iteration + 1

                response = await self._get_response(messages)

                run.total_input_tokens += response.input_tokens
                run.total_cached_tokens += response.cached_tokens
                run.total_output_tokens += response.output_tokens
                run.total_reasoning_tokens += response.reasoning_tokens
                run.total_latency_ms += response.latency_ms

                if response.tool_calls:
                    await self._process_tool_calls(
                        response, messages, run
                    )
                else:
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
        context: dict[str, Any] | None = None,
    ) -> list[Message]:
        messages = [
            Message(role="system", content=self._build_system_prompt()),
        ]

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
        if not self.tools:
            return self.system_prompt

        tool_descriptions = "\n".join(
            f"- **{tool.name}**: {tool.description}" for tool in self.tools
        )

        full_prompt = f"{self.system_prompt}\n\n## Available Tools\n{tool_descriptions}"
        logger.debug(f"System prompt with {len(self.tools)} tools prepared")

        return full_prompt

    async def _get_response(self, messages: list[Message]) -> ProviderResponse:
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

        for tool_call in response.tool_calls or []:
            run.tool_calls_count += 1

            action_step = AgentStep(
                step_type=StepType.ACTION,
                content=f"Calling {tool_call.name}",
                tool_name=tool_call.name,
                tool_input=tool_call.arguments,
                latency_ms=response.latency_ms,
            )
            run.steps.append(action_step)

            logger.debug(f"Executing tool: {tool_call.name} with args: {json.dumps(tool_call.arguments, indent=2)}")

            start_time = time.time()
            try:
                tool_result = await self._execute_tool(
                    tool_call.name, tool_call.arguments
                )
            except Exception as e:
                tool_result = json.dumps({"error": str(e)})
                logger.error(f"Tool execution failed: {e}")

            execution_time = (time.time() - start_time) * 1000

            context_result, csv_path = self._result_processor.process_result(
                tool_call.name, tool_result
            )

            log_preview = tool_result[:TOOL_RESULT_PREVIEW_LENGTH] + "..." if len(tool_result) > TOOL_RESULT_PREVIEW_LENGTH else tool_result
            logger.debug(f"Tool {tool_call.name} result (truncated): {log_preview}")
            if csv_path:
                logger.info(f"Large result saved to CSV: {csv_path}")

            obs_step = AgentStep(
                step_type=StepType.OBSERVATION,
                content=context_result,
                tool_name=tool_call.name,
                tool_output=tool_result,
                latency_ms=execution_time,
            )
            run.steps.append(obs_step)

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
        """Create a tool message, handling images from RAG results."""
        images = self._extract_images_from_result(result)

        if not images:
            return Message(
                role="tool",
                content=context_result,
                tool_call_id=tool_call_id,
                name=tool_name,
            )

        content_parts: list[TextContent | ImageContent] = [TextContent(text=context_result)]

        for img in images:
            content_parts.append(
                ImageContent(
                    image_base64=img.get("base64", img.get("image_base64", "")),
                    media_type=img.get("media_type", "image/jpeg"),
                )
            )

        logger.debug(f"Tool {tool_name} returned {len(images)} image(s)")

        return Message(
            role="tool",
            content=context_result,
            content_parts=content_parts,
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    def _extract_images_from_result(self, result: str) -> list[dict[str, str]]:
        """Extract base64 images from RAG search results."""
        return self._result_processor.extract_images(result)

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        if tool_name not in self._tool_registry:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        result = self.tool_executor(tool_name, arguments)

        if hasattr(result, "__await__"):
            result = await result

        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)
        return str(result)

    def _default_tool_executor(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Default tool executor that returns an error."""
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
        self._provider: BaseProvider | None = None
        self._tools: list[ToolDefinition] = []
        self._tool_executor: ToolExecutor | None = None
        self._max_iterations: int = MAX_ITERATIONS
        self._system_prompt: str | None = None
        self._csv_threshold: int = CSV_THRESHOLD
        self._csv_output_dir: str = "./agent_outputs"

    def with_provider(self, provider: BaseProvider) -> Self:
        """Set the LLM provider."""
        self._provider = provider
        return self

    def with_tool(self, tool: ToolDefinition) -> Self:
        """Add a tool to the agent."""
        self._tools.append(tool)
        return self

    def with_tools(self, tools: list[ToolDefinition]) -> Self:
        """Add multiple tools to the agent."""
        self._tools.extend(tools)
        return self

    def with_tool_executor(self, executor: ToolExecutor) -> Self:
        """Set the tool executor function."""
        self._tool_executor = executor
        return self

    def with_max_iterations(self, max_iterations: int) -> Self:
        """Set the maximum number of iterations."""
        self._max_iterations = max_iterations
        return self

    def with_system_prompt(self, prompt: str) -> Self:
        """Set a custom system prompt."""
        self._system_prompt = prompt
        return self

    def with_csv_threshold(self, threshold: int) -> Self:
        """Set the row count threshold for saving results to CSV."""
        self._csv_threshold = threshold
        return self

    def with_csv_output_dir(self, directory: str) -> Self:
        """Set the directory for CSV output files."""
        self._csv_output_dir = directory
        return self

    def with_config(self, config: AgentConfig) -> Self:
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
