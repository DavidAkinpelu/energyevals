"""JSON file observer for local observability.

This module provides a file-based observer that writes complete agent traces
to JSON files. Useful for local development, debugging, and offline analysis.

Features:
- Complete trace data (no truncation)
- All steps including failed tool calls
- Full tool inputs and outputs
- Error details and stack traces
- Configurable output formats (individual files or JSONL)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from energbench.agent.schema import AgentRun, AgentStep, StepType
from .base import BaseObserver


class JSONFileObserver(BaseObserver):
    """Observer that writes agent runs to local JSON files.

    Captures complete trace data including:
    - Full query and response
    - All execution steps (action, observation, answer, error, thought)
    - Complete tool inputs and outputs (not truncated)
    - Failed tool calls with error details
    - Token usage and latency metrics
    - Timestamps for each step

    Output Formats:
    - Individual files: One JSON file per trace (easier to inspect)
    - JSONL: All traces in one file, one JSON object per line (easier to process)
    """

    def __init__(
        self,
        output_dir: str = "./observability_logs",
        single_file: bool = False,
        filename: str = "agent_traces.jsonl",
        pretty_print: bool = True,
        include_raw_messages: bool = True,
    ):
        """Initialize the JSON observer.

        Args:
            output_dir: Directory to store trace files.
            single_file: If True, append all traces to one JSONL file.
                        If False, create one JSON file per trace.
            filename: Filename for single_file mode (should end in .jsonl).
            pretty_print: If True, format JSON with indentation (individual files only).
            include_raw_messages: If True, include raw LLM message history when available.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.single_file = single_file
        self.filename = filename
        self.pretty_print = pretty_print
        self.include_raw_messages = include_raw_messages
        self._enabled = True

        logger.info(f"JSONFileObserver initialized. Output dir: {self.output_dir}")

    @property
    def is_enabled(self) -> bool:
        """Check if the observer is enabled."""
        return self._enabled

    def trace_agent_run(
        self,
        run: AgentRun,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Write complete agent run to JSON file.

        Captures ALL data from the run including:
        - Every step (no filtering)
        - Full tool outputs (no truncation)
        - Error details for failed steps
        - Complete metrics

        Args:
            run: The AgentRun to trace.
            metadata: Additional metadata to attach.
            tags: Tags for categorizing the trace.
            user_id: User identifier.
            session_id: Session identifier.

        Returns:
            Trace ID (timestamp-based unique identifier).
        """
        if not self._enabled:
            return None

        try:
            trace_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            trace_data = self._build_trace_data(
                trace_id=trace_id,
                run=run,
                metadata=metadata,
                tags=tags,
                user_id=user_id,
                session_id=session_id,
            )

            self._write_trace(trace_id, trace_data)

            logger.debug(f"Traced agent run to JSON: {trace_id}")
            return trace_id

        except Exception as e:
            logger.error(f"Failed to write trace to JSON: {e}")
            return None

    def _build_trace_data(
        self,
        trace_id: str,
        run: AgentRun,
        metadata: Optional[dict[str, Any]],
        tags: Optional[list[str]],
        user_id: Optional[str],
        session_id: Optional[str],
    ) -> dict[str, Any]:
        """Build complete trace data structure."""

        # Analyze steps for summary
        step_summary = self._analyze_steps(run.steps)

        return {
            # Identification
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "start_time": datetime.fromtimestamp(run.start_time).isoformat() if run.start_time else None,
            "end_time": datetime.fromtimestamp(run.end_time).isoformat() if run.end_time else None,

            # Query and Response
            "query": run.query,
            "final_answer": run.final_answer,

            # Status
            "success": run.success,
            "error": run.error,

            # Metrics - Complete token breakdown
            "metrics": {
                "iterations": run.iterations,
                "tool_calls_count": run.tool_calls_count,
                "total_input_tokens": run.total_input_tokens,
                "total_output_tokens": run.total_output_tokens,
                "total_cached_tokens": run.total_cached_tokens,
                "total_tokens": run.total_tokens,
                "total_latency_ms": run.total_latency_ms,
                "duration_seconds": run.duration_seconds,
            },

            # Step Summary
            "step_summary": step_summary,

            # All Steps - Complete and unfiltered
            "steps": [
                self._serialize_step(step, index)
                for index, step in enumerate(run.steps)
            ],

            # Metadata
            "metadata": metadata or {},
            "tags": tags or [],
            "user_id": user_id,
            "session_id": session_id,
        }

    def _analyze_steps(self, steps: list[AgentStep]) -> dict[str, Any]:
        """Analyze steps and provide summary statistics."""
        summary = {
            "total_steps": len(steps),
            "step_types": {},
            "tool_calls": [],
            "failed_tool_calls": [],
            "errors": [],
        }

        for step in steps:
            # Count step types
            step_type = step.step_type.value
            summary["step_types"][step_type] = summary["step_types"].get(step_type, 0) + 1

            # Track tool calls
            if step.step_type == StepType.ACTION and step.tool_name:
                summary["tool_calls"].append(step.tool_name)

            # Track failed tool calls (check observation for errors)
            if step.step_type == StepType.OBSERVATION and step.tool_output:
                if self._is_tool_error(step.tool_output):
                    summary["failed_tool_calls"].append({
                        "tool": step.tool_name,
                        "error_preview": step.tool_output[:200] if len(step.tool_output) > 200 else step.tool_output,
                    })

            # Track error steps
            if step.step_type == StepType.ERROR:
                summary["errors"].append(step.content)

        return summary

    def _is_tool_error(self, output: str) -> bool:
        """Check if tool output indicates an error."""
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                # Check for common error patterns
                if data.get("error"):
                    return True
                if data.get("success") is False:
                    return True
                if "error" in str(data).lower() and "errno" not in str(data).lower():
                    return True
        except (json.JSONDecodeError, TypeError):
            pass
        return False

    def _serialize_step(self, step: AgentStep, index: int) -> dict[str, Any]:
        """Serialize a single step with complete data."""
        step_data = {
            "index": index,
            "step_type": step.step_type.value,
            "timestamp": datetime.fromtimestamp(step.timestamp).isoformat() if step.timestamp else None,
            "timestamp_unix": step.timestamp,

            # Content - never truncated
            "content": step.content,

            # Tool information (for action/observation steps)
            "tool_name": step.tool_name,
            "tool_input": step.tool_input,
            "tool_output": step.tool_output,  # Full output, never truncated
            "tool_output_length": len(step.tool_output) if step.tool_output else 0,

            # Metrics
            "tokens_used": step.tokens_used,
            "latency_ms": step.latency_ms,
        }

        # Add error analysis for observation steps
        if step.step_type == StepType.OBSERVATION and step.tool_output:
            step_data["is_error"] = self._is_tool_error(step.tool_output)
            if step_data["is_error"]:
                step_data["error_details"] = self._extract_error_details(step.tool_output)

        return step_data

    def _extract_error_details(self, output: str) -> Optional[dict[str, Any]]:
        """Extract error details from tool output."""
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                return {
                    "error_message": data.get("error"),
                    "error_type": data.get("error_type"),
                    "success": data.get("success"),
                }
        except (json.JSONDecodeError, TypeError):
            return {"raw_error": output[:500]}
        return None

    def _write_trace(self, trace_id: str, trace_data: dict[str, Any]) -> None:
        """Write trace data to file."""
        if self.single_file:
            # Append to JSONL file (one JSON object per line)
            filepath = self.output_dir / self.filename
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_data, default=str, ensure_ascii=False) + "\n")
            logger.debug(f"Appended trace to {filepath}")
        else:
            # Write individual JSON file
            filepath = self.output_dir / f"trace_{trace_id}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                if self.pretty_print:
                    json.dump(trace_data, f, indent=2, default=str, ensure_ascii=False)
                else:
                    json.dump(trace_data, f, default=str, ensure_ascii=False)
            logger.debug(f"Wrote trace to {filepath}")

    def flush(self) -> None:
        """No-op for file observer (writes are immediate)."""
        pass

    def shutdown(self) -> None:
        """Shutdown the observer."""
        self._enabled = False
        logger.info("JSONFileObserver shutdown")

    def get_trace_file(self, trace_id: str) -> Optional[Path]:
        """Get the file path for a specific trace.

        Args:
            trace_id: The trace ID to look up.

        Returns:
            Path to the trace file, or None if not found.
        """
        if self.single_file:
            return self.output_dir / self.filename
        else:
            filepath = self.output_dir / f"trace_{trace_id}.json"
            return filepath if filepath.exists() else None

    def list_traces(self) -> list[str]:
        """List all trace IDs in the output directory.

        Returns:
            List of trace IDs.
        """
        if self.single_file:
            filepath = self.output_dir / self.filename
            if not filepath.exists():
                return []
            traces = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        traces.append(data.get("trace_id", "unknown"))
                    except json.JSONDecodeError:
                        continue
            return traces
        else:
            return [
                f.stem.replace("trace_", "")
                for f in self.output_dir.glob("trace_*.json")
            ]

    def load_trace(self, trace_id: str) -> Optional[dict[str, Any]]:
        """Load a specific trace by ID.

        Args:
            trace_id: The trace ID to load.

        Returns:
            Trace data dict, or None if not found.
        """
        if self.single_file:
            filepath = self.output_dir / self.filename
            if not filepath.exists():
                return None
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get("trace_id") == trace_id:
                            return data
                    except json.JSONDecodeError:
                        continue
            return None
        else:
            filepath = self.output_dir / f"trace_{trace_id}.json"
            if not filepath.exists():
                return None
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
