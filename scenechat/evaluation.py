"""Deterministic quality metrics for generated scenarios and simulation traces.

The evaluator deliberately avoids another model call: it turns structural
invariants and observable trace properties into repeatable scores that can be
used in local experiments or CI without changing production behavior.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from math import log
from typing import Iterable

from .models import Message, SimulationState
from .scenario import ScenarioPackage, validate_scenario_package


@dataclass(frozen=True)
class QualityScore:
    value: float
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def _score(value: float, detail: str, threshold: float = 0.8) -> QualityScore:
    bounded = max(0.0, min(1.0, value))
    return QualityScore(round(bounded, 4), bounded >= threshold, detail)


def evaluate_scenario(package: ScenarioPackage) -> dict[str, QualityScore]:
    """Score constraint fidelity, privacy, identity, and runtime readiness."""
    constraints = [item for item in package.brief.constraints if item.locked and item.content]
    covered = set(package.world.covered_constraint_ids)
    for fact in package.world.facts:
        covered.update(fact.covered_constraint_ids)
    for character in package.characters:
        covered.update(character.covered_constraint_ids)
    covered_count = sum(item.id in covered for item in constraints)

    requested = package.brief.requested_character_count
    actual = len(package.characters)
    count_value = 1.0 if requested is None else max(0.0, 1 - abs(actual - requested) / max(requested, 1))
    names = [character.name for character in package.characters]
    identity_value = len(set(names)) / max(len(names), 1)

    validation_issues = validate_scenario_package(package)
    privacy_issues = [issue for issue in validation_issues if "私密约束" in issue]
    runtime_issues = [
        issue for issue in validation_issues
        if any(marker in issue for marker in ("规则", "阶段", "effect", "状态操作", "结束条件"))
    ]
    structured = len(package.world.phases) > 1 or package.world.scheduler != "round_robin"
    runtime_parts = [bool(package.world.opening_scene), bool(package.world.termination_conditions)]
    if structured:
        runtime_parts.extend([
            bool(package.world.phase_specs),
            bool(package.world.rules),
            bool(package.world.termination_rules),
        ])
    runtime_value = sum(runtime_parts) / len(runtime_parts)
    if runtime_issues:
        runtime_value *= 0.5

    return {
        "constraint_recall": _score(
            covered_count / max(len(constraints), 1),
            f"{covered_count}/{len(constraints)} locked constraints declared covered",
        ),
        "character_count": _score(count_value, f"requested={requested}, actual={actual}", 1.0),
        "unique_identity": _score(identity_value, f"{len(set(names))}/{len(names)} unique names", 1.0),
        "visibility_safety": _score(1.0 if not privacy_issues else 0.0, "; ".join(privacy_issues) or "no private constraint found in public text", 1.0),
        "runtime_completeness": _score(runtime_value, "; ".join(runtime_issues) or "required runtime structures present"),
        "deterministic_validation": _score(1.0 if not validation_issues else 0.0, "; ".join(validation_issues) or "package valid", 1.0),
    }


def _near_duplicate_rate(speeches: list[str]) -> float:
    if len(speeches) < 2:
        return 0.0
    duplicates = 0
    comparisons = 0
    for index, speech in enumerate(speeches):
        normalized = "".join(speech.split()).lower()
        for previous in speeches[max(0, index - 5):index]:
            comparisons += 1
            other = "".join(previous.split()).lower()
            if normalized and other and SequenceMatcher(None, normalized, other).ratio() >= 0.88:
                duplicates += 1
                break
    return duplicates / max(comparisons, 1)


def evaluate_trace(
    messages: Iterable[Message],
    *,
    expected_characters: Iterable[str] = (),
    state: SimulationState | None = None,
    expect_end: bool = False,
) -> dict[str, QualityScore]:
    """Score only public, user-observable properties of a simulation trace."""
    trace = list(messages)
    dialogue = [message for message in trace if message.kind != "narration" and message.speech.strip()]
    speeches = [message.speech for message in dialogue]
    duplicate_rate = _near_duplicate_rate(speeches)
    counts = Counter(message.speaker for message in dialogue)
    expected = list(expected_characters)
    participation = len(set(counts).intersection(expected)) / max(len(expected), 1) if expected else (1.0 if counts else 0.0)

    if len(counts) <= 1:
        balance = 0.0 if len(dialogue) > 1 else 1.0
    else:
        total = sum(counts.values())
        entropy = -sum((count / total) * log(count / total) for count in counts.values())
        balance = entropy / log(len(counts))

    role_samples: dict[str, str] = {}
    for message in dialogue:
        role_samples.setdefault(message.speaker, "")
        role_samples[message.speaker] += message.speech
    pairs = []
    sample_values = list(role_samples.values())
    for index, sample in enumerate(sample_values):
        for other in sample_values[index + 1:]:
            pairs.append(SequenceMatcher(None, sample[:800], other[:800]).ratio())
    differentiation = 1 - (sum(pairs) / len(pairs)) if pairs else (1.0 if len(sample_values) == 1 else 0.0)

    ended = bool(state and state.ended)
    natural_end = ended and bool(state.end_reason) and state.end_kind not in {"max_turns", "blocked"}
    failure_count = state.failed_generation_count if state else 0
    blocked = bool(state and state.run_status == "blocked")

    metrics = {
        "non_empty_dialogue": _score(1.0 if dialogue else 0.0, f"{len(dialogue)} dialogue events", 1.0),
        "duplicate_avoidance": _score(1 - duplicate_rate, f"near-duplicate rate={duplicate_rate:.3f}"),
        "cast_participation": _score(participation, f"{len(set(counts).intersection(expected))}/{len(expected)} expected characters spoke" if expected else f"{len(counts)} speakers"),
        "turn_balance": _score(balance, f"normalized speaker entropy={balance:.3f}", 0.65),
        "role_differentiation": _score(differentiation, f"lexical differentiation={differentiation:.3f}", 0.45),
        "generation_stability": _score(0.0 if blocked else 1 / (1 + failure_count), f"failed generations={failure_count}, blocked={blocked}", 0.8),
    }
    if expect_end or ended:
        metrics["natural_ending"] = _score(1.0 if natural_end else 0.0, f"ended={ended}, kind={getattr(state, 'end_kind', '')}", 1.0)
    return metrics


def summarize_quality(metrics: dict[str, QualityScore]) -> dict:
    values = [metric.value for metric in metrics.values()]
    return {
        "score": round(sum(values) / max(len(values), 1), 4),
        "passed": all(metric.passed for metric in metrics.values()),
        "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
    }
