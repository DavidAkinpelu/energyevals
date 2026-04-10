import ast
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from loguru import logger

from .base_tool import BaseTool, tool_method
from .constants import SYSTEM_COMMAND_TIMEOUT, SYSTEM_MAX_RESULTS

_ALLOWED_COMMANDS = {
    "awk",
    "cat",
    "cut",
    "date",
    "df",
    "du",
    "echo",
    "find",
    "grep",
    "head",
    "ls",
    "pwd",
    "python",
    "python3",
    "rg",
    "sed",
    "sort",
    "stat",
    "tail",
    "uniq",
    "wc",
}
_BLOCKED_IMPORTS = {
    "ctypes",
    "multiprocessing",
    "socket",
    "subprocess",
}
_BLOCKED_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "input",
}
_BLOCKED_ATTR_CALLS: dict[str, set[str]] = {
    "os": {"system", "popen", "fork", "kill", "remove", "unlink", "rmdir", "rename", "chmod", "chown"},
    "pathlib": {"rmdir", "unlink", "rename", "replace", "chmod", "chown"},
}
_DEFAULT_ALLOWED_ROOTS = (
    Path.cwd().resolve(),
    Path("/tmp").resolve(),
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"Invalid {name}={raw!r}; using default {default}.")
        return default
    if value <= 0:
        logger.warning(f"Non-positive {name}={raw!r}; using default {default}.")
        return default
    return value


_PY_SANDBOX_TIMEOUT_SECONDS = _env_int("ENERGBENCH_PY_SANDBOX_TIMEOUT_SECONDS", 300)
_PY_SANDBOX_MEM_BYTES = _env_int("ENERGBENCH_PY_SANDBOX_MEM_BYTES", 4 * 1024 * 1024 * 1024)
_PY_SANDBOX_FILE_BYTES = _env_int("ENERGBENCH_PY_SANDBOX_FILE_BYTES", 50 * 1024 * 1024)
_PY_SANDBOX_RUNNER = r"""
import builtins
import json
import os
import pathlib
import resource
import shlex
import socket
import subprocess
import sys
import traceback

ALLOWED_ROOTS = [pathlib.Path(p).resolve() for p in json.loads(sys.argv[1])]
CPU_SECONDS = int(sys.argv[2])
MEM_BYTES = int(sys.argv[3])
FILE_BYTES = int(sys.argv[4])
# Allow solver binaries used by pyomo/cvxpy while keeping subprocess access narrow.
ALLOWED_EXECUTABLES = {"ipopt", "highs", "glpsol", "cbc"}


def _within_allowed(path: object) -> bool:
    if not isinstance(path, (str, bytes, os.PathLike)):
        return True
    resolved = pathlib.Path(path).expanduser().resolve()
    for root in ALLOWED_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _blocked(*_args, **_kwargs):
    raise PermissionError("Operation blocked by sandbox policy")


def _extract_executable(command):
    if isinstance(command, (list, tuple)):
        if not command:
            return None
        cmd = command[0]
    elif isinstance(command, (str, bytes)):
        text = command.decode() if isinstance(command, bytes) else command
        parts = shlex.split(text)
        if not parts:
            return None
        cmd = parts[0]
    elif isinstance(command, os.PathLike):
        cmd = os.fspath(command)
    else:
        return None
    return os.fspath(cmd)


def _is_allowed_subprocess_command(command) -> bool:
    executable = _extract_executable(command)
    if not executable:
        return False
    executable_name = pathlib.Path(executable).name
    return executable_name in ALLOWED_EXECUTABLES


def _guard_subprocess_args(*args, **kwargs):
    if kwargs.get("shell"):
        raise PermissionError("subprocess shell execution is blocked by sandbox policy")
    command = kwargs.get("args", args[0] if args else None)
    if not _is_allowed_subprocess_command(command):
        raise PermissionError("subprocess command is blocked by sandbox policy")
    return command


resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))

_real_open = builtins.open


def _sandbox_open(file, *args, **kwargs):
    if not _within_allowed(file):
        raise PermissionError(f"Path outside sandbox roots: {file}")
    return _real_open(file, *args, **kwargs)


builtins.open = _sandbox_open
_real_popen = subprocess.Popen
_real_run = subprocess.run
_real_call = subprocess.call
_real_check_call = subprocess.check_call
_real_check_output = subprocess.check_output


def _guarded_popen(*args, **kwargs):
    _guard_subprocess_args(*args, **kwargs)
    return _real_popen(*args, **kwargs)


def _guarded_run(*args, **kwargs):
    _guard_subprocess_args(*args, **kwargs)
    return _real_run(*args, **kwargs)


def _guarded_call(*args, **kwargs):
    _guard_subprocess_args(*args, **kwargs)
    return _real_call(*args, **kwargs)


def _guarded_check_call(*args, **kwargs):
    _guard_subprocess_args(*args, **kwargs)
    return _real_check_call(*args, **kwargs)


def _guarded_check_output(*args, **kwargs):
    _guard_subprocess_args(*args, **kwargs)
    return _real_check_output(*args, **kwargs)


subprocess.Popen = _guarded_popen
subprocess.run = _guarded_run
subprocess.call = _guarded_call
subprocess.check_call = _guarded_check_call
subprocess.check_output = _guarded_check_output
socket.socket = _blocked
os.system = _blocked
os.popen = _blocked
os.fork = _blocked

user_code = sys.stdin.read()
# Use one namespace for globals/locals so imports are visible inside functions/classes
# defined by user code. Separate dicts make top-level imports local-only, which then
# raises NameError inside function bodies.
exec_namespace = {"__name__": "__main__"}

try:
    exec(compile(user_code, "<sandbox>", "exec"), exec_namespace, exec_namespace)
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""


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
        try:
            self._validate_python_code(code)
        except Exception as exc:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Sandbox policy violation: {exc}",
                    "stdout": "",
                    "stderr": "",
                },
                indent=2,
            )

        allowed_roots = [str(path) for path in _DEFAULT_ALLOWED_ROOTS]

        try:
            python_executable = os.environ.get("PYTHON_EXECUTABLE")
            if not python_executable:
                python_executable = "python3" if shutil.which("python3") else "python"

            result = subprocess.run(
                [
                    python_executable,
                    "-I",
                    "-c",
                    _PY_SANDBOX_RUNNER,
                    json.dumps(allowed_roots),
                    str(_PY_SANDBOX_TIMEOUT_SECONDS),
                    str(_PY_SANDBOX_MEM_BYTES),
                    str(_PY_SANDBOX_FILE_BYTES),
                ],
                input=code,
                capture_output=True,
                text=True,
                timeout=_PY_SANDBOX_TIMEOUT_SECONDS,
                check=False,
                cwd=str(Path.cwd()),
            )
            if result.returncode == 0:
                return json.dumps(
                    {
                        "status": "success",
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                    indent=2,
                )

            stderr = result.stderr.strip()
            err_line = stderr.splitlines()[-1] if stderr else f"Sandbox execution failed ({result.returncode})"
            return json.dumps(
                {
                    "status": "error",
                    "error": err_line,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                indent=2,
            )
        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Execution timed out after {_PY_SANDBOX_TIMEOUT_SECONDS}s",
                    "stdout": "",
                    "stderr": "",
                },
                indent=2,
            )
        except Exception as exc:
            return json.dumps({"status": "error", "error": str(exc), "stdout": "", "stderr": ""}, indent=2)

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
            if not args:
                return json.dumps({"status": "error", "error": "Empty command"}, indent=2)

            command_name = Path(args[0]).name
            if command_name not in _ALLOWED_COMMANDS:
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            f"Command '{command_name}' is blocked by sandbox policy. "
                            f"Allowed commands: {', '.join(sorted(_ALLOWED_COMMANDS))}"
                        ),
                    },
                    indent=2,
                )

            safe_cwd = self._resolve_cwd(cwd)
            safe_timeout = max(1, min(timeout, SYSTEM_COMMAND_TIMEOUT))
            logger.info(f"Running sandboxed command: {' '.join(args)} (cwd={safe_cwd})")

            result = subprocess.run(
                args,
                cwd=str(safe_cwd),
                capture_output=True,
                text=True,
                timeout=safe_timeout,
                check=False,
                env=self._sandbox_env(),
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

    def _resolve_cwd(self, cwd: str | None) -> Path:
        candidate = (Path(cwd) if cwd else Path.cwd()).expanduser().resolve()
        for root in _DEFAULT_ALLOWED_ROOTS:
            if self._is_relative_to(candidate, root):
                return candidate
        raise ValueError(
            f"Working directory '{candidate}' is outside sandbox roots: "
            f"{', '.join(str(root) for root in _DEFAULT_ALLOWED_ROOTS)}"
        )

    def _sandbox_env(self) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        }
        return env

    def _validate_python_code(self, code: str) -> None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in _BLOCKED_IMPORTS:
                        raise ValueError(f"Import of module '{root}' is not allowed")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".", 1)[0]
                    if root in _BLOCKED_IMPORTS:
                        raise ValueError(f"Import from module '{root}' is not allowed")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
                    raise ValueError(f"Call to '{node.func.id}' is not allowed")
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    module_name = node.func.value.id
                    attr_name = node.func.attr
                    if attr_name in _BLOCKED_ATTR_CALLS.get(module_name, set()):
                        raise ValueError(f"Call to '{module_name}.{attr_name}' is not allowed")

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
