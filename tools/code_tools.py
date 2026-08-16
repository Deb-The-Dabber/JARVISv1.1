import ast
import os
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", "venv", "node_modules", ".venv"}
COUNT_EXTENSIONS = {".py", ".js", ".html", ".md"}


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def read_file(path: str, offset: int = 0) -> str:
    """Read a file and return up to 200 lines starting at the given offset."""
    try:
        p = _expand(path)
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        offset = int(offset)
        start = offset
        end = offset + 200
        shown = lines[start:end]
        result = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(shown, start=start))
        if len(lines) > end:
            result += f"\n... truncated: showing lines {start + 1}-{end} of {len(lines)}."
        return result or f"(empty from offset {offset})"
    except Exception as e:
        return f"Could not read file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    try:
        p = _expand(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {p}."
    except Exception as e:
        return f"Could not write file: {e}"


def append_file(path: str, content: str) -> str:
    """Append content to an existing or new file."""
    try:
        p = _expand(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} characters to {p}."
    except Exception as e:
        return f"Could not append file: {e}"


def list_directory(path: str = "~") -> str:
    """List visible files and directories."""
    try:
        p = _expand(path)
        if not p.exists():
            return f"Path does not exist: {p}"
        if not p.is_dir():
            return f"Not a directory: {p}"
        rows = []
        for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if item.name.startswith("."):
                continue
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            rows.append(f"{prefix} {item.name}")
        return "\n".join(rows) if rows else "(empty directory)"
    except Exception as e:
        return f"Could not list directory: {e}"


def run_python(code: str) -> str:
    """Run Python code in a subprocess with a 30 second timeout."""
    from sandbox import run_sandboxed_python

    result = run_sandboxed_python(code, timeout=30)
    parts = [f"Exit code: {result['exit_code']}"]
    stdout = result.get("stdout", "").strip()
    stderr = result.get("stderr", "").strip()
    if stdout:
        parts.append(f"STDOUT:\n{stdout}")
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    if result.get("error"):
        parts.append(f"Error: {result['error']}")
    if result.get("sandboxed"):
        parts.append("[Sandboxed]")
    return "\n\n".join(parts)


def scan_project_structure(path: str = "~/Jarvis") -> str:
    """Walk a project up to 3 levels deep and summarize files."""
    try:
        root = _expand(path)
        if not root.exists():
            return f"Path does not exist: {root}"

        lines = [str(root)]
        total_files = 0
        total_lines = 0

        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            rel = current_path.relative_to(root)
            depth = 0 if str(rel) == "." else len(rel.parts)
            if depth >= 3:
                dirs[:] = []
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

            indent = "  " * depth
            if depth > 0:
                lines.append(f"{indent}[DIR] {current_path.name}/")

            file_indent = "  " * (depth + 1)
            for name in sorted(files):
                if name.startswith("."):
                    continue
                file_path = current_path / name
                try:
                    size = file_path.stat().st_size
                except Exception:
                    size = 0
                total_files += 1
                suffix = file_path.suffix.lower()
                line_count = ""
                if suffix in COUNT_EXTENSIONS:
                    try:
                        count = len(file_path.read_text(encoding="utf-8", errors="ignore").splitlines())
                        total_lines += count
                        line_count = f", {count} lines"
                    except Exception:
                        pass
                lines.append(f"{file_indent}[FILE] {name} ({size} bytes{line_count})")

        lines.append(f"\nTotal files: {total_files}")
        lines.append(f"Total counted lines: {total_lines}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not scan project: {e}"


def search_in_files(query: str, path: str = "~/Jarvis", extension: str = ".py") -> str:
    """Search for a string across matching files."""
    try:
        root = _expand(path)
        matches = []
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for name in files:
                if not name.endswith(extension):
                    continue
                file_path = Path(current) / name
                try:
                    for line_no, line in enumerate(file_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if query in line:
                            matches.append(f"{file_path}:{line_no}: {line.strip()}")
                            if len(matches) >= 20:
                                return "\n".join(matches)
                except Exception:
                    continue
        return "\n".join(matches) if matches else "No matches found."
    except Exception as e:
        return f"Search failed: {e}"


def get_function_signatures(path: str) -> str:
    """Extract function and class definitions from a Python file."""
    try:
        p = _expand(path)
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        lines = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                if node.args.vararg:
                    args.append("*" + node.args.vararg.arg)
                args.extend(a.arg for a in node.args.kwonlyargs)
                if node.args.kwarg:
                    args.append("**" + node.args.kwarg.arg)
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                lines.append(f"{p}:{node.lineno}: {prefix} {node.name}({', '.join(args)})")
            elif isinstance(node, ast.ClassDef):
                lines.append(f"{p}:{node.lineno}: class {node.name}")
        return "\n".join(sorted(lines, key=lambda s: int(s.split(":")[1]))) if lines else "No functions or classes found."
    except Exception as e:
        return f"Could not parse signatures: {e}"


CODE_TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "list_directory": list_directory,
    "run_python": run_python,
    "scan_project_structure": scan_project_structure,
    "search_in_files": search_in_files,
    "get_function_signatures": get_function_signatures,
}


CODE_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return the first 200 numbered lines, with a truncation notice if longer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."},
                    "offset": {"type": "integer", "description": "Line number to start reading from (0-indexed). Use to paginate through large files."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating parent directories if needed.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append content to an existing or new file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List visible files and directories with [DIR] and [FILE] prefixes.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory to list. Defaults to ~."}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python code in a subprocess with a 30 second timeout and return stdout, stderr, and exit code.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_project_structure",
            "description": "Walk a project up to 3 levels deep, skipping common dependency/cache directories, and report file sizes and line counts.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Project path. Defaults to ~/Jarvis."}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "Search for a query string across files with a matching extension and return up to 20 filename:line matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "description": "Root path to search. Defaults to ~/Jarvis."},
                    "extension": {"type": "string", "description": "File extension to search. Defaults to .py."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_function_signatures",
            "description": "Use Python AST parsing to list class and function definitions with line numbers.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Python file path to inspect."}}, "required": ["path"]},
        },
    },
]
