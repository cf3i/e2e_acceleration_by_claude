"""The agent: a per-task state machine, independent of how generation is scheduled.

    prompt_text() -> engine generates -> observe(text) -> (optional tool call -> add_tool_result) -> ...
"""
import re
from dataclasses import dataclass, field

from .tools import TOOLS

SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step. "
    "Use the python tool for any non-trivial arithmetic (always print() the result). "
    "When you know the final answer, state it as \\boxed{N} where N is a plain number."
)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([\w\-]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([\w\-]+)>\n?(.*?)\n?</parameter>", re.S)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class AgentState:
    task: dict
    max_turns: int = 6
    messages: list = field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0
    done: bool = False
    answer: str | None = None
    finish_reason: str = ""

    def __post_init__(self):
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.task["question"]},
        ]

    def prompt_text(self, tok) -> str:
        return tok.apply_chat_template(
            self.messages, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def observe(self, text: str):
        """Consume one assistant turn. Returns (tool_name, args) if a tool must run, else None."""
        self.turns += 1
        self.messages.append({"role": "assistant", "content": text})
        m = TOOL_CALL_RE.search(text)
        if m:
            if self.turns >= self.max_turns:
                self.done, self.finish_reason = True, "max_turns"
                return None
            self.tool_calls += 1
            return m.group(1), dict(PARAM_RE.findall(m.group(2)))
        boxed = BOXED_RE.findall(text)
        self.answer = boxed[-1] if boxed else None
        self.done, self.finish_reason = True, "answer" if boxed else "no_answer"
        return None

    def add_tool_result(self, result: str):
        self.messages.append({"role": "tool", "content": result})

    @property
    def correct(self) -> bool:
        return is_correct(self.answer, self.task["answer"])


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    nums = NUM_RE.findall(pred.replace(",", ""))
    if not nums:
        return False
    try:
        p, g = float(nums[-1]), float(gold)
    except ValueError:
        return False
    return abs(p - g) <= 1e-4 * max(1.0, abs(g))
