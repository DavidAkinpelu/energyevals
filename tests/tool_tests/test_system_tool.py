import json

import pytest

from energyevals.tools.system_tool import SystemTool


class TestSystemToolInit:
    """Tests for SystemTool initialization."""

    def test_init(self):
        """Test basic initialization."""
        tool = SystemTool()
        assert tool.name == "system"
        assert "filesystem" in tool.description.lower() or "command" in tool.description.lower()

    def test_get_tools_definition(self):
        """Test tool definitions."""
        tool = SystemTool()
        tools = tool.get_tools()

        assert len(tools) == 4
        tool_names = {t.name for t in tools}
        assert "list_files" in tool_names
        assert "grep_files" in tool_names
        assert "run_python_code" in tool_names
        assert "run_shell_command" in tool_names


class TestListFiles:
    """Tests for list_files method."""

    def test_list_files_in_existing_directory(self, tmp_path):
        """Test listing files in an existing directory."""
        (tmp_path / "file1.txt").write_text("test")
        (tmp_path / "file2.py").write_text("code")

        tool = SystemTool()
        result = tool.list_files(path=str(tmp_path))

        data = json.loads(result)
        assert "results" in data
        assert data["count"] >= 2

    def test_list_files_recursive(self, tmp_path):
        """Test recursive file listing."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root")
        (subdir / "nested.txt").write_text("nested")

        tool = SystemTool()
        result = tool.list_files(path=str(tmp_path), recursive=True)

        data = json.loads(result)
        assert data["count"] >= 2
        results_str = "\n".join(data["results"])
        assert "nested.txt" in results_str

    def test_list_files_nonrecursive(self, tmp_path):
        """Test non-recursive listing."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root")
        (subdir / "nested.txt").write_text("nested")

        tool = SystemTool()
        result = tool.list_files(path=str(tmp_path), recursive=False)

        data = json.loads(result)
        assert data["count"] == 2

    def test_list_files_nonexistent_path(self):
        """Test listing files from nonexistent path."""
        tool = SystemTool()
        result = tool.list_files(path="/nonexistent/path")

        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_list_files_max_results(self, tmp_path):
        """Test max_results limit."""
        for i in range(50):
            (tmp_path / f"file{i}.txt").write_text(f"content{i}")

        tool = SystemTool()
        result = tool.list_files(path=str(tmp_path), max_results=10)

        data = json.loads(result)
        assert data["count"] == 10


class TestGrepFiles:
    """Tests for grep_files method."""

    def test_grep_simple_pattern(self, tmp_path):
        """Test searching for a simple pattern."""
        file1 = tmp_path / "test1.txt"
        file1.write_text("Hello world\nFoo bar\n")
        file2 = tmp_path / "test2.txt"
        file2.write_text("Hello again\nBaz qux\n")

        tool = SystemTool()
        result = tool.grep_files(pattern="Hello", path=str(tmp_path))

        data = json.loads(result)
        assert data["count"] >= 2
        results_str = "\n".join(data["results"])
        assert "Hello" in results_str

    def test_grep_case_insensitive(self, tmp_path):
        """Test case-insensitive search."""
        file1 = tmp_path / "test.txt"
        file1.write_text("UPPERCASE\nlowercase\n")

        tool = SystemTool()
        result = tool.grep_files(pattern="uppercase", path=str(tmp_path), case_insensitive=True)

        data = json.loads(result)
        assert data["count"] >= 1

    def test_grep_with_glob_filter(self, tmp_path):
        """Test grep with glob pattern filter."""
        (tmp_path / "test.py").write_text("print('hello')")
        (tmp_path / "test.txt").write_text("hello")

        tool = SystemTool()
        result = tool.grep_files(pattern="hello", path=str(tmp_path), glob="*.py")

        data = json.loads(result)
        results_str = "\n".join(data["results"])
        assert ".py" in results_str
        assert ".txt" not in results_str or data["count"] == 1

    def test_grep_nonexistent_path(self):
        """Test grep on nonexistent path."""
        tool = SystemTool()
        result = tool.grep_files(pattern="test", path="/nonexistent")

        data = json.loads(result)
        assert "error" in data

    def test_grep_max_results(self, tmp_path):
        """Test max_results limit in grep."""
        file1 = tmp_path / "test.txt"
        lines = "\n".join([f"match {i}" for i in range(100)])
        file1.write_text(lines)

        tool = SystemTool()
        result = tool.grep_files(pattern="match", path=str(tmp_path), max_results=10)

        data = json.loads(result)
        assert data["count"] == 10


class TestRunPythonCode:
    """Tests for run_python_code method."""

    def test_run_simple_python_code(self):
        """Test running simple Python code."""
        tool = SystemTool()
        result = tool.run_python_code("print('hello')")

        data = json.loads(result)
        assert data["status"] == "success"
        assert "hello" in data["stdout"]

    def test_run_python_code_with_calculation(self):
        """Test running Python code with calculations."""
        tool = SystemTool()
        code = "result = 2 + 2\nprint(f'Result: {result}')"
        result = tool.run_python_code(code)

        data = json.loads(result)
        assert data["status"] == "success"
        assert "4" in data["stdout"]

    def test_run_python_code_with_error(self):
        """Test running Python code that raises an error."""
        tool = SystemTool()
        result = tool.run_python_code("raise ValueError('test error')")

        data = json.loads(result)
        assert data["status"] == "error"
        assert "test error" in data["error"]

    def test_run_python_code_captures_stderr(self):
        """Test that stderr is captured."""
        tool = SystemTool()
        code = "import sys\nprint('error message', file=sys.stderr)"
        result = tool.run_python_code(code)

        data = json.loads(result)
        assert "error message" in data["stderr"]

    def test_run_python_code_namespace_isolation(self):
        """Test that code runs in isolated namespace."""
        tool = SystemTool()

        result1 = tool.run_python_code("x = 10\nprint(x)")
        data1 = json.loads(result1)
        assert "10" in data1["stdout"]

        result2 = tool.run_python_code("print(x)")
        data2 = json.loads(result2)
        assert data2["status"] == "error"
        assert "not defined" in data2["error"]

    def test_run_python_code_scientific_imports(self):
        """Test that common scientific packages can be imported in sandbox."""
        tool = SystemTool()
        code = """
import numpy
import pandas
import scipy
import cvxpy
print("OK")
"""
        result = tool.run_python_code(code)

        data = json.loads(result)
        assert data["status"] == "success"
        assert "OK" in data["stdout"]

    def test_run_python_code_import_scope_inside_function(self):
        """Test that module imports are visible inside function scope."""
        tool = SystemTool()
        code = """
import numpy as np

def compute():
    return int(np.sum(np.array([1, 2, 3])))

print(compute())
"""
        result = tool.run_python_code(code)

        data = json.loads(result)
        assert data["status"] == "success"
        assert "6" in data["stdout"]

    def test_run_python_code_pyomo_ipopt_path(self):
        """Test that pyomo import and optional ipopt checks run in sandbox."""
        tool = SystemTool()
        code = """
from pyomo.environ import ConcreteModel, NonNegativeReals, Objective, Var, maximize
from pyomo.opt import SolverFactory

model = ConcreteModel()
model.x = Var(domain=NonNegativeReals, bounds=(0, 1))
model.obj = Objective(expr=model.x, sense=maximize)
solver = SolverFactory("ipopt")
if solver is None or not solver.available():
    print("IPOPT_UNAVAILABLE")
else:
    solver.solve(model)
    print("IPOPT_SOLVED")
"""
        result = tool.run_python_code(code)
        data = json.loads(result)
        assert data["status"] == "success"
        assert "IPOPT_UNAVAILABLE" in data["stdout"] or "IPOPT_SOLVED" in data["stdout"]


class TestRunShellCommand:
    """Tests for run_shell_command method."""

    def test_run_simple_shell_command(self):
        """Test running simple shell command."""
        tool = SystemTool()
        result = tool.run_shell_command("echo hello")

        data = json.loads(result)
        assert data["status"] == "success"
        assert "hello" in data["stdout"]
        assert data["returncode"] == 0

    def test_run_command_with_working_directory(self, tmp_path):
        """Test running command with specific working directory."""
        tool = SystemTool()
        result = tool.run_shell_command("pwd", cwd=str(tmp_path))

        data = json.loads(result)
        assert data["status"] == "success"
        assert str(tmp_path) in data["stdout"]

    def test_run_command_that_fails(self):
        """Test running command that fails."""
        tool = SystemTool()
        result = tool.run_shell_command("ls /nonexistent_directory_12345")

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["returncode"] != 0

    def test_run_command_captures_stderr(self):
        """Test that stderr is captured."""
        tool = SystemTool()
        result = tool.run_shell_command("ls /nonexistent 2>&1 || echo error")

        data = json.loads(result)
        assert data["status"] == "success"

    def test_run_invalid_command(self):
        """Test running invalid command."""
        tool = SystemTool()
        result = tool.run_shell_command("nonexistent_command_xyz")

        data = json.loads(result)
        assert "error" in data or data["returncode"] != 0

    def test_run_python_command(self):
        """Test running python command."""
        tool = SystemTool()
        result = tool.run_shell_command('python -c "print(123)"')

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["returncode"] == 0
        assert "123" in data["stdout"]

    def test_run_python3_command(self):
        """Test running python3 command."""
        tool = SystemTool()
        result = tool.run_shell_command('python3 -c "print(456)"')

        data = json.loads(result)
        assert data["status"] == "success"
        assert data["returncode"] == 0
        assert "456" in data["stdout"]


@pytest.mark.integration
class TestSystemToolIntegration:
    """Integration tests for SystemTool with real filesystem operations."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow: create files, list, grep, execute."""
        tool = SystemTool()

        create_code = f"""
import pathlib
p = pathlib.Path('{tmp_path}')
(p / 'script.py').write_text('print("test")')
(p / 'data.txt').write_text('important data')
"""
        result1 = tool.run_python_code(create_code)
        data1 = json.loads(result1)
        assert data1["status"] == "success"

        result2 = tool.list_files(path=str(tmp_path))
        data2 = json.loads(result2)
        assert data2["count"] >= 2

        result3 = tool.grep_files(pattern="important", path=str(tmp_path))
        data3 = json.loads(result3)
        assert data3["count"] >= 1
