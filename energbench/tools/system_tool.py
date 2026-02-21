import io
import json
import re
import shlex
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from loguru import logger

from .base_tool import BaseTool, tool_method
from .constants import SYSTEM_COMMAND_TIMEOUT, SYSTEM_MAX_RESULTS


class SystemTool(BaseTool):
    """Tooling for local file search and command execution."""

    def __init__(self) -> None:
        super().__init__(
            name="system",
            description="Local filesystem and command utilities",
        )

    @tool_method()
    def list_files(
        self,
        path: str = ".",
        recursive: bool = False,
        max_results: int = SYSTEM_MAX_RESULTS,
    ) -> str:
        """List files and directories under a given path, returning their full paths.

        Args:
            path: Path to list files from (defaults to current directory).
            recursive: Whether to list files recursively through subdirectories.
            max_results: Maximum number of results to return.

        Returns:
            JSON string with path, count, and a list of file/directory paths.
        """
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

    @tool_method()
    def grep_files(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        case_insensitive: bool = False,
        max_results: int = SYSTEM_MAX_RESULTS,
    ) -> str:
        """Search files for a pattern using ripgrep if available, falling back to Python regex.
        Returns matching lines in file:line:content format.

        Args:
            pattern: Regex pattern to search for.
            path: Path to search (defaults to current directory).
            glob: Optional glob filter (e.g., '*.py').
            case_insensitive: Whether to search case-insensitively.
            max_results: Maximum number of results to return.

        Returns:
            JSON string with count and a list of matching lines.
        """
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

    @tool_method()
    def run_python_code(self, code: str) -> str:
        """Execute Python code in the current process and return stdout/stderr.
        The code runs with access to all installed packages (pandas, numpy, etc.) and any data
        loaded in the current session.

        Args:
            code: Python code to execute. Has access to all installed packages (pandas, numpy, etc.).

        Returns:
            JSON string with status, stdout, and stderr.
        """
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

    @tool_method()
    def run_shell_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = SYSTEM_COMMAND_TIMEOUT,
    ) -> str:
        """Run a shell command locally and return stdout/stderr.

        Args:
            command: Shell command to run.
            cwd: Working directory. Defaults to the current working directory.
            timeout: Timeout in seconds.

        Returns:
            JSON string with status, returncode, stdout, and stderr.
        """
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
