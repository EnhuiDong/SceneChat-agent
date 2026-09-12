"""Conservative UTF-8 byte budgets, not claims of provider token accuracy."""
from .config import config_int
from .errors import SceneChatError


def size(text):
    return len(text.encode("utf-8"))


def enforce(prompt):
    maximum = config_int("simulation", "input_budget_bytes", 96000, minimum=4000, maximum=2000000)
    if size(prompt) > maximum:
        raise SceneChatError("input_budget_exceeded", "必要设定与状态超过输入预算。请提高 config.json 中 simulation.input_budget_bytes 或精简设定；未删除硬约束。", stage="simulation", status_code=422)
    return prompt


def optional(text, maximum):
    lines, seen, used = [], set(), 0
    for line in text.splitlines():
        if line in seen:
            continue
        seen.add(line)
        cost = size(line) + 1
        if used + cost > maximum:
            continue
        lines.append(line)
        used += cost
    return "\n".join(lines)
