"""Deterministic quality metrics for generated scenarios and simulation traces.

The evaluator deliberately avoids another model call: it turns structural
invariants and observable trace properties into repeatable scores that can be
used in local experiments or CI without changing production behavior.
"""

from __future__ import annotations

import re
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
    obligations = {
        message.event_id
        for message in dialogue
        if isinstance(message.intent, dict)
        and message.intent.get("conversation_move") in {"question", "request", "challenge"}
        and message.intent.get("addressed_to")
    }
    replies = {
        str(message.intent.get("reply_to_event_id") or "")
        for message in dialogue
        if isinstance(message.intent, dict) and message.intent.get("reply_to_event_id")
    }
    responsiveness = len(obligations.intersection(replies)) / max(len(obligations), 1)
    threaded = sum(
        bool(
            message.intent.get("thread_id")
            or message.intent.get("reply_to_event_id")
            or message.intent.get("mentioned_agents")
        )
        for message in dialogue
        if isinstance(message.intent, dict)
    ) / max(len(dialogue), 1)
    monologues = sum(
        len(message.speech) > 240
        or len([part for part in re.split(r"[。！？!?；;]+", message.speech) if part.strip()]) > 4
        for message in dialogue
    )
    relationship_updates = [
        update
        for message in dialogue
        if isinstance(message.intent, dict)
        for update in (message.intent.get("relationship_updates") or {}).values()
        if isinstance(update, dict)
    ]
    evidenced_relationships = sum(
        bool(update.get("reason_event_id") and update.get("private_note"))
        for update in relationship_updates
    )

    metrics = {
        "non_empty_dialogue": _score(1.0 if dialogue else 0.0, f"{len(dialogue)} dialogue events", 1.0),
        "duplicate_avoidance": _score(1 - duplicate_rate, f"near-duplicate rate={duplicate_rate:.3f}"),
        "cast_participation": _score(participation, f"{len(set(counts).intersection(expected))}/{len(expected)} expected characters spoke" if expected else f"{len(counts)} speakers"),
        "turn_balance": _score(balance, f"normalized speaker entropy={balance:.3f}", 0.65),
        "role_differentiation": _score(differentiation, f"lexical differentiation={differentiation:.3f}", 0.45),
        "generation_stability": _score(0.0 if blocked else 1 / (1 + failure_count), f"failed generations={failure_count}, blocked={blocked}", 0.8),
        "conversation_responsiveness": _score(
            responsiveness if obligations else 1.0,
            f"answered obligations={len(obligations.intersection(replies))}/{len(obligations)}",
            0.75,
        ),
        "threaded_interaction": _score(
            min(1.0, threaded * 2) if dialogue else 0.0,
            f"linked or mention-aware turns={threaded:.3f}",
            0.4,
        ),
        "monologue_avoidance": _score(
            1 - monologues / max(len(dialogue), 1),
            f"overlong dialogue turns={monologues}/{len(dialogue)}",
        ),
        "relationship_evidence": _score(
            evidenced_relationships / max(len(relationship_updates), 1)
            if relationship_updates else 1.0,
            f"evidenced updates={evidenced_relationships}/{len(relationship_updates)}",
            1.0,
        ),
    }
    if state is not None:
        thread_obligations = [
            obligation
            for thread in state.conversation_threads.values()
            for obligation in thread.obligations
        ]
        if thread_obligations:
            responded_obligations = sum(
                item.status in {"responded", "satisfied", "withdrawn"}
                for item in thread_obligations
            )
            metrics["conversation_responsiveness"] = _score(
                responded_obligations / len(thread_obligations),
                f"thread obligations responded={responded_obligations}/{len(thread_obligations)}",
                0.75,
            )
        active_beliefs = [
            belief
            for agent in state.agents.values()
            for belief in agent.belief_records
            if belief.active
        ]
        sourced_beliefs = sum(
            bool(item.source_event_id)
            or item.epistemic_status in {"inferred", "believed"}
            for item in active_beliefs
        )
        metrics["belief_provenance"] = _score(
            sourced_beliefs / max(len(active_beliefs), 1) if active_beliefs else 1.0,
            f"traceable beliefs={sourced_beliefs}/{len(active_beliefs)}",
            0.9,
        )
        evidence_keys = []
        for agent in state.agents.values():
            for target, relationship in agent.relationship_dynamics.items():
                for evidence in relationship.get("evidence") or []:
                    if not isinstance(evidence, dict):
                        continue
                    for facet in (evidence.get("facets") or {}):
                        evidence_keys.append(
                            (agent.name, target, str(evidence.get("event_id") or ""), facet)
                        )
        unique_evidence = len(set(evidence_keys))
        metrics["relationship_evidence_reuse"] = _score(
            unique_evidence / max(len(evidence_keys), 1) if evidence_keys else 1.0,
            f"unique event/facet applications={unique_evidence}/{len(evidence_keys)}",
            1.0,
        )
        structured_memories = sum(
            memory.memory_type != "event"
            for agent in state.agents.values()
            for memory in agent.memories
        )
        reflection_noise = sum(
            memory.source == "private_reflection"
            for agent in state.agents.values()
            for memory in agent.memories
        )
        metrics["memory_signal"] = _score(
            structured_memories / max(structured_memories + reflection_noise, 1)
            if structured_memories or reflection_noise else 1.0,
            f"structured memories={structured_memories}, reflection noise={reflection_noise}",
            0.8,
        )
        quality_retries = getattr(state, "dialogue_quality_retry_count", 0)
        metrics["quality_gate_stability"] = _score(
            1 / (1 + quality_retries / max(len(dialogue), 1)),
            f"quality retries={quality_retries}/{len(dialogue)} dialogue turns",
            0.7,
        )
    if expect_end or ended:
        metrics["natural_ending"] = _score(1.0 if natural_end else 0.0, f"ended={ended}, kind={getattr(state, 'end_kind', '')}", 1.0)
    return metrics


def evaluate_director_control(state: SimulationState) -> dict[str, QualityScore]:
    """Score persisted intervention safety and observable application results."""

    interventions = list(state.interventions)
    ids = [item.id for item in interventions]
    unique_ids = len(ids) == len(set(ids))
    valid_statuses = {"preview", "pending", "applied", "cancelled", "rejected"}
    lifecycle_issues = [
        item.id for item in interventions
        if item.status not in valid_statuses
        or (item.status == "applied" and item.applied_at_turn is None)
    ]

    timeline_ids: set[str] = set()
    private_event_ids = set()
    unsafe_private_patches = []
    for message in state.history:
        if isinstance(message.intent, dict):
            timeline_ids.update(
                str(item) for item in message.intent.get("intervention_ids") or []
            )
            intervention_id = message.intent.get("intervention_id")
            if intervention_id:
                timeline_ids.add(str(intervention_id))
        if message.visibility == "audience_only":
            private_event_ids.add(message.event_id)
            if message.state_patch or message.state_updates:
                unsafe_private_patches.append(message.event_id)

    missing_timeline = [
        item.id for item in interventions
        if item.status == "applied" and item.id not in timeline_ids
    ]
    leaked_private_events = []
    for agent in state.agents.values():
        memory_event_ids = {memory.event_id for memory in agent.memories}
        leaked_private_events.extend(sorted(private_event_ids & memory_event_ids))

    blocking_applied = [
        item.id for item in interventions
        if item.status in {"pending", "applied"}
        and item.mode != "override"
        and any(conflict.get("severity") == "blocking" for conflict in item.conflicts)
    ]
    patch_issues = [
        item.id for item in interventions
        if (item.mode == "guidance" and item.proposed_patch)
        or (item.visibility == "audience_only" and item.proposed_patch)
    ]

    return {
        "intervention_identity": _score(
            1.0 if unique_ids else 0.0,
            "all intervention ids unique" if unique_ids else "duplicate intervention ids found",
            1.0,
        ),
        "lifecycle_integrity": _score(
            1.0 if not lifecycle_issues else 0.0,
            "statuses and applied turns are valid"
            if not lifecycle_issues else f"invalid lifecycle: {lifecycle_issues}",
            1.0,
        ),
        "timeline_accounting": _score(
            1.0 if not missing_timeline else 0.0,
            "all applied interventions are traceable"
            if not missing_timeline else f"missing timeline markers: {missing_timeline}",
            1.0,
        ),
        "director_privacy": _score(
            1.0 if not leaked_private_events and not unsafe_private_patches else 0.0,
            "reader-only events stayed out of agent memory and public state"
            if not leaked_private_events and not unsafe_private_patches
            else f"leaked={leaked_private_events}, unsafe_patches={unsafe_private_patches}",
            1.0,
        ),
        "conflict_enforcement": _score(
            1.0 if not blocking_applied else 0.0,
            "no blocked non-override intervention was queued or applied"
            if not blocking_applied else f"blocked interventions active: {blocking_applied}",
            1.0,
        ),
        "patch_boundary": _score(
            1.0 if not patch_issues else 0.0,
            "guidance and reader-only interventions carry no state patch"
            if not patch_issues else f"unsafe intervention patches: {patch_issues}",
            1.0,
        ),
    }


def compare_pacing_runs(
    slow_state: SimulationState,
    fast_state: SimulationState,
) -> dict[str, QualityScore]:
    """Compare two runs of the same scenario using observable pacing effects."""

    slow_events = max(len(slow_state.history), 1)
    fast_events = max(len(fast_state.history), 1)
    slow_density = slow_state.narration_count / slow_events
    fast_density = fast_state.narration_count / fast_events
    slow_remaining = max(
        0,
        (slow_state.arc_state.target_end_turn or slow_state.turn_count)
        - slow_state.turn_count,
    )
    fast_remaining = max(
        0,
        (fast_state.arc_state.target_end_turn or fast_state.turn_count)
        - fast_state.turn_count,
    )

    return {
        "target_horizon_order": _score(
            1.0 if fast_remaining < slow_remaining else 0.0,
            f"slow remaining={slow_remaining}, fast remaining={fast_remaining}",
            1.0,
        ),
        "narration_density_order": _score(
            1.0 if fast_density >= slow_density else 0.0,
            f"slow={slow_density:.3f}, fast={fast_density:.3f}",
            1.0,
        ),
        "progress_order": _score(
            1.0
            if fast_state.ended or fast_state.arc_state.progress > slow_state.arc_state.progress
            else 0.0,
            f"slow={slow_state.arc_state.progress:.3f}, fast={fast_state.arc_state.progress:.3f}",
            1.0,
        ),
    }


def summarize_quality(metrics: dict[str, QualityScore]) -> dict:
    values = [metric.value for metric in metrics.values()]
    return {
        "score": round(sum(values) / max(len(values), 1), 4),
        "passed": all(metric.passed for metric in metrics.values()),
        "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
    }
