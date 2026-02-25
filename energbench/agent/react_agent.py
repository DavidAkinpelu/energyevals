import asyncio
import concurrent.futures
import inspect
import json
import re
import time
from typing import Any

from loguru import logger

from .constants import (
    CSV_THRESHOLD,
    MAX_ITERATIONS,
    MAX_TOOL_RESULT_CHARS,
    PROVIDER_MAX_RETRIES,
    PROVIDER_RETRY_BASE_DELAY,
    QUERY_TRUNCATE_LENGTH,
    TOOL_TIMEOUT,
)
from energbench.core.retry import retry_with_backoff

from .exceptions import ProviderError, ToolExecutionError
from .processors import ResultProcessor
from .prompts import get_system_prompt
from .providers import BaseProvider, ProviderResponse, ToolDefinition
from .schema import (
    AgentRun,
    AgentStep,
    ImageContent,
    Message,
    StepType,
    TextContent,
    ToolExecutor,
)

_RAW_TOOL_CALL_RE = re.compile(r"<function=\w+.*?</function>", re.DOTALL)


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
        tool_timeout: float = TOOL_TIMEOUT,
        max_retries: int = PROVIDER_MAX_RETRIES,
        retry_base_delay: float = PROVIDER_RETRY_BASE_DELAY,
        max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
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
            tool_timeout: Seconds before a stalled tool call is cancelled (default: 60).
            max_retries: Maximum retries for provider complete() on transient errors (default: 3).
            retry_base_delay: Base delay in seconds for exponential backoff (default: 1.0).
            max_tool_result_chars: Truncate tool results to this many chars before adding to LLM context (0 = disabled).
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
        self.tool_timeout = tool_timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.max_tool_result_chars = max_tool_result_chars

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
                    should_continue = await self._process_tool_calls(
                        response, messages, run, iteration
                    )
                    if not should_continue:
                        break
                else:
                    if response.content and _RAW_TOOL_CALL_RE.search(response.content):
                        logger.warning(
                            "Model returned raw text tool call(s) instead of structured tool_calls. "
                            "The provider may not have parsed them. "
                            f"Content preview: {response.content[:200]}"
                        )
                    run.final_answer = response.content
                    run.steps.append(
                        AgentStep(
                            step_type=StepType.ANSWER,
                            content=response.content,
                            iteration=iteration,
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
        return await self._retry_complete(
            messages=messages,
            tools=self.tools if self.tools else None,
        )

    async def _retry_complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
    ) -> ProviderResponse:
        total_attempts = 1 + self.max_retries

        async def on_retry(attempt: int, exc: Exception, delay: float) -> None:
            logger.warning(
                f"Provider call failed (attempt {attempt + 1}/{total_attempts}), "
                f"retrying in {delay:.1f}s: {exc}"
            )
            await asyncio.sleep(delay)

        try:
            return await retry_with_backoff(
                lambda: self.provider.complete(messages, tools=tools, temperature=0.0),
                max_retries=self.max_retries,
                base_delay=self.retry_base_delay,
                on_retry=on_retry,
            )
        except Exception as exc:
            raise ProviderError(str(exc), provider=self.provider.provider_name) from exc

    async def _process_tool_calls(
        self,
        response: ProviderResponse,
        messages: list[Message],
        run: AgentRun,
        iteration: int = 0,
    ) -> bool:
        """Process tool calls from the provider response.

        Returns:
            True if the agent loop should continue, False if a non-recoverable
            tool error requires stopping immediately.
        """
        messages.append(
            Message(
                role="assistant",
                content=response.content,
                tool_calls=[
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "thought_signature": tc.thought_signature,
                    }
                    for tc in response.tool_calls
                ]
                if response.tool_calls
                else None,
            )
        )

        # Log the model's reasoning as a THOUGHT step (owns the LLM latency + tokens).
        run.steps.append(
            AgentStep(
                step_type=StepType.THOUGHT,
                content=response.content or "",
                iteration=iteration,
                tokens_used=response.input_tokens + response.output_tokens,
                latency_ms=response.latency_ms,
            )
        )

        llm_call_timestamp = time.time()

        for tool_call in response.tool_calls or []:
            run.tool_calls_count += 1

            action_step = AgentStep(
                step_type=StepType.ACTION,
                content=f"Calling {tool_call.name}",
                iteration=iteration,
                tool_name=tool_call.name,
                tool_input=tool_call.arguments,
                # Latency belongs to the THOUGHT step (the LLM call); tool
                # execution latency is captured on the OBSERVATION step.
                latency_ms=0.0,
                timestamp=llm_call_timestamp,
            )
            run.steps.append(action_step)

            logger.debug(f"Executing tool: {tool_call.name} with args: {json.dumps(tool_call.arguments, indent=2)}")

            start_time = time.time()
            try:
                tool_result = await self._execute_tool(
                    tool_call.name, tool_call.arguments
                )
            except TimeoutError:
                error_payload = {"error": f"Tool '{tool_call.name}' timed out after {self.tool_timeout}s"}
                logger.error(error_payload["error"])
                tool_result = json.dumps(error_payload)
            except Exception as e:
                error_payload = {"error": str(e), "tool": tool_call.name, "error_type": type(e).__name__}
                logger.error(f"Tool '{tool_call.name}' failed: {e}", exc_info=True)
                tool_result = json.dumps(error_payload)

            execution_time = (time.time() - start_time) * 1000

            # Check for non-recoverable tool failure before processing further.
            try:
                result_data = json.loads(tool_result)
                if (
                    isinstance(result_data, dict)
                    and not result_data.get("success", True)
                    and result_data.get("metadata", {}).get("recoverable") is False
                ):
                    error_msg = result_data.get("error") or f"Non-recoverable error in tool '{tool_call.name}'"
                    logger.error(f"Non-recoverable tool error: {error_msg}")
                    run.success = False
                    run.error = error_msg
                    run.steps.append(AgentStep(step_type=StepType.ERROR, content=error_msg))
                    return False
            except json.JSONDecodeError:
                pass  # Not valid JSON or unexpected shape — treat as recoverable

            context_result, csv_path = self._result_processor.process_result(
                tool_call.name, tool_result
            )

            if self.max_tool_result_chars > 0 and len(context_result) > self.max_tool_result_chars:
                logger.warning(
                    f"Tool {tool_call.name} result truncated from {len(context_result)} "
                    f"to {self.max_tool_result_chars} chars before adding to context"
                )
                context_result = (
                    context_result[: self.max_tool_result_chars]
                    + f"\n...[truncated: result exceeded {self.max_tool_result_chars} chars]"
                )

            logger.debug(f"Tool {tool_call.name} returned {len(tool_result)} chars")
            if csv_path:
                logger.info(f"Large result saved to CSV: {csv_path}")

            obs_step = AgentStep(
                step_type=StepType.OBSERVATION,
                content=context_result,
                iteration=iteration,
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

        return True

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
            raise ToolExecutionError(f"Unknown tool: {tool_name}", tool_name=tool_name)

        if inspect.iscoroutinefunction(self.tool_executor):
            result = await asyncio.wait_for(
                self.tool_executor(tool_name, arguments),
                timeout=self.tool_timeout,
            )
        else:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, self.tool_executor, tool_name, arguments),
                    timeout=self.tool_timeout,
                )
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=self.tool_timeout)

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
