from __future__ import annotations

import io
import json
import re
import shlex
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from loguru import logger

from energbench.agent.providers import ToolDefinition

from .base_tool import BaseTool
from .constants import SYSTEM_COMMAND_TIMEOUT, SYSTEM_MAX_RESULTS


class SystemTool(BaseTool):
    """Tooling for local file search and command execution."""

    def __init__(self) -> None:
        super().__init__(
            name="system",
            description="Local filesystem and command utilities",
        )
        self.register_method("list_files", self.list_files)
        self.register_method("grep_files", self.grep_files)
        self.register_method("run_python_code", self.run_python_code)
        self.register_method("run_shell_command", self.run_shell_command)

    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="list_files",
                description="List files under a path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to list files from"},
                        "recursive": {"type": "boolean", "default": False},
                        "max_results": {"type": "integer", "default": SYSTEM_MAX_RESULTS},
                    },
                },
            ),
            ToolDefinition(
                name="grep_files",
                description="Search files for a pattern (uses rg if available).",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "path": {"type": "string", "description": "Path to search"},
                        "glob": {"type": "string", "description": "Optional glob filter (e.g., '*.py')"},
                        "case_insensitive": {"type": "boolean", "default": False},
                        "max_results": {"type": "integer", "default": SYSTEM_MAX_RESULTS},
                    },
                    "required": ["pattern"],
                },
            ),
            ToolDefinition(
                name="run_python_code",
                description="Execute Python code in-process and return stdout/stderr.",
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                    },
                    "required": ["code"],
                },
            ),
            ToolDefinition(
                name="run_shell_command",
                description="Run a shell command locally.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run"},
                        "cwd": {"type": "string", "description": "Working directory"},
                        "timeout": {"type": "integer", "default": SYSTEM_COMMAND_TIMEOUT},
                    },
                    "required": ["command"],
                },
            ),
        ]

    def list_files(
        self,
        path: str = ".",
        recursive: bool = False,
        max_results: int = SYSTEM_MAX_RESULTS,
    ) -> str:
        try:
            base = Path(path).expanduser()
            if not base.exists():
                return json.dumps({"error": f"Path not found: {path}"})

            results: list[str] = []
            if recursive:
                for entry in base.rglob("*"):
                    results.append(str(entry))
                    if len(results) >= max_results:
                        break
            else:
                for entry in base.iterdir():
                    results.append(str(entry))
                    if len(results) >= max_results:
                        break

            return json.dumps({"path": str(base), "count": len(results), "results": results}, indent=2)
        except Exception as exc:
            logger.error(f"list_files failed: {exc}")
            return json.dumps({"error": str(exc)})

    def grep_files(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        case_insensitive: bool = False,
        max_results: int = SYSTEM_MAX_RESULTS,
    ) -> str:
        base = Path(path).expanduser()
        if not base.exists():
            return json.dumps({"error": f"Path not found: {path}"})

        rg_cmd = ["rg", "--no-messages", "--line-number"]
        if case_insensitive:
            rg_cmd.append("-i")
        if glob:
            rg_cmd.extend(["-g", glob])
        rg_cmd.extend([pattern, str(base)])

        try:
            completed = subprocess.run(
                rg_cmd,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode in {0, 1}:
                lines = completed.stdout.strip().splitlines()
                if max_results:
                    lines = lines[:max_results]
                return json.dumps({"count": len(lines), "results": lines}, indent=2)
        except FileNotFoundError:
            logger.debug("rg not available, falling back to Python search.")
        except Exception as exc:
            logger.error(f"rg search failed: {exc}")

        results: list[str] = []
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern, flags)

        paths = base.rglob("*") if base.is_dir() else [base]
        for file_path in paths:
            if not file_path.is_file():
                continue
            if glob and not file_path.match(glob):
                continue
            try:
                for idx, line in enumerate(file_path.read_text(errors="ignore").splitlines(), start=1):
                    if regex.search(line):
                        results.append(f"{file_path}:{idx}:{line}")
                        if len(results) >= max_results:
                            break
            except Exception:
                continue
            if len(results) >= max_results:
                break

        return json.dumps({"count": len(results), "results": results}, indent=2)

    def run_python_code(self, code: str) -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        namespace: dict[str, object] = {}
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, namespace)
            return json.dumps(
                {
                    "status": "success",
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                },
                indent=2,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "status": "error",
                    "error": str(exc),
                    "stdout": stdout.getvalue(),
                    "stderr": stderr.getvalue(),
                },
                indent=2,
            )

    def run_shell_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = SYSTEM_COMMAND_TIMEOUT,
    ) -> str:
        try:
            args = shlex.split(command)
            result = subprocess.run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return json.dumps(
                {
                    "status": "success",
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                indent=2,
            )
        except Exception as exc:
            return json.dumps({"status": "error", "error": str(exc)}, indent=2)
