"""Tools available to the agent. Stage 00: a fresh python subprocess per call."""
import subprocess
import sys

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute a Python 3 script and return its stdout. Use print() to show results.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source code to execute."}},
                "required": ["code"],
            },
        },
    }
]


def run_python(code: str, timeout: float = 10.0, max_chars: int = 1000) -> str:
    try:
        p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=timeout)
        out = p.stdout
        if p.returncode != 0:
            out += "\n" + p.stderr[-500:]
    except subprocess.TimeoutExpired:
        out = "Error: execution timed out"
    out = out.strip() or "(no output; use print() to show results)"
    return out[:max_chars]


def execute(name: str, args: dict) -> str:
    if name == "python" and "code" in args:
        return run_python(args["code"])
    return f"Error: unknown tool or missing arguments: {name}({', '.join(args)})"
