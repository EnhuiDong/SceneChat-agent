"""Bounded recovery shared by build, simulation and intervention paths."""
import hashlib
import json
import re
import time
from functools import wraps

from .build_control import BuildControl, CURRENT_BUILD
from .config import config_int
from .errors import SceneChatError


# Bump only when stalled structured or semantic checkpoints gain a genuinely
# new repair policy. Older failed builds may then receive one bounded retry epoch.
REPAIR_POLICY_VERSION = 7


def upgraded_repair_available(row):
    if not row or row.get("status") != "failed":
        return False
    last_error = row.get("payload", {}).get("last_error") or {}
    if last_error.get("resumable") is not False:
        return False
    if (last_error.get("error") or {}).get("code") not in {"scenario_repair_stalled", "structured_output_stalled"}:
        return False
    try:
        previous = int(last_error.get("recovery_version") or 0)
    except (TypeError, ValueError):
        previous = 0
    return previous < REPAIR_POLICY_VERSION


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     default=str).encode("utf-8")).hexdigest()


def is_transient(exc):
    # Wrapped provider failures retain their cause; cancellation/configuration
    # errors must never turn into another paid request.
    if isinstance(exc, SceneChatError):
        return (exc.code.endswith(("_connection_failed", "_unavailable"))
                and exc.cause is not None and is_transient(exc.cause))
    text = str(exc).lower()
    if any(marker in text for marker in (
        "allocationquota", "insufficient", "quota exhausted", "authentication",
        "unauthorized", "invalid api key", "invalid_api_key", "invalidapikey",
        "model not found", "invalid model", "free tier only",
    )):
        return False
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in {408, 409, 429} or isinstance(status, int) and 500 <= status < 600
    return isinstance(exc, (TimeoutError, ConnectionError)) or any(
        marker in (type(exc).__name__ + " " + text).lower()
        for marker in ("timeout", "timed out", "apiconnectionerror", "connection reset",
                       "connection error", "temporarily unavailable", "bad gateway",
                       "service unavailable", "gateway timeout")
    )


def retry_delay(control, attempt):
    # Short, cancellation-aware backoff. Deadline and cancellation are checked
    # before *every* new request, including a retry after a successful sleep.
    end = time.monotonic() + min(2, 0.5 * 2 ** attempt)
    while time.monotonic() < end:
        control.check()
        control.cancelled.wait(min(0.05, max(0, end - time.monotonic())))
    control.check()


def transport_call(operation, *, control, retries, limit=None):
    for attempt in range(retries + 1):
        control.check()
        if limit is not None:
            counter = getattr(control, "request_owner", control)
            if counter.step_requests >= limit:
                raise SceneChatError("model_request_budget_exceeded",
                                     "本次操作已达到请求上限，已停止自动重试；已提交内容保留。",
                                     stage=control.stage, status_code=409)
            counter.step_requests += 1
        try:
            result = operation()
            control.check()
            return result
        except Exception as exc:
            if attempt >= retries or not is_transient(exc):
                raise
            control.progress(reason="服务暂时不可用，正在进行有次数及时间限制的重试",
                             transport_retry=attempt + 1)
            retry_delay(control, attempt)


def bounded_operation(stage):
    """Nested actor/narrator/verifier requests share one event's wall budget."""
    def decorate(operation):
        @wraps(operation)
        def wrapped(*args, **kwargs):
            if CURRENT_BUILD.get() is not None:
                return operation(*args, **kwargs)
            control = BuildControl()
            control.stage = stage
            control.deadline = control.step_deadline = time.monotonic() + config_int(
                "simulation", "operation_timeout_seconds", 180, minimum=5, maximum=600)
            with control.activate():
                state = args[0] if args else kwargs.get("state")
                scheduler_index = getattr(state, "_scheduler_index", None)
                try:
                    return operation(*args, **kwargs)
                except Exception:
                    # Failed generation has not committed an event. A retry
                    # must select the same actor, not silently skip their turn.
                    if operation.__name__ == "simulate_next_event" and scheduler_index is not None:
                        state._scheduler_index = scheduler_index
                    raise
        return wrapped
    return decorate


def repair_local_fields(package):
    """Repair only unambiguous serialization mistakes, never invent mechanics."""
    from .scenario_patch import apply_scenario_patch
    changes = []
    for index, rule in enumerate(package.world.rules):
        for effect_index, effect in enumerate(rule.effects):
            if (effect.op == "set_phase" and effect.value in (None, "")
                    and effect.target in package.world.phases):
                changes.append({"path": f"/world/rules/{index}/effects/{effect_index}/value",
                                "value": effect.target})
    names = {character.name for character in package.characters if character.name}
    aliases = {
        character.id: character.name for character in package.characters
        if character.id and character.name and character.id != character.name
        and sum(other.id == character.id for other in package.characters) == 1
    }
    dimensions = {item.id for item in package.world.relationship_dimensions}
    if not dimensions:
        dimensions = {"cooperation", "confidence", "regard"}
    dimension_aliases = {
        "cooperation_potential": "cooperation",
        "trust": "confidence",
    }
    for index, character in enumerate(package.characters):
        hidden_identity = (
            bool(character.faction or character.private_identity)
            and any(token in package.world.public_world_markdown
                    for token in ("隐藏身份", "身份隐藏", "秘密身份", "伪装"))
        )
        if hidden_identity:
            for field_name in ("public_identity", "public_traits", "public_background"):
                public_value = str(getattr(character, field_name) or "")
                leak = re.search(r"(?:实际是|其实是|真实身份是|隐藏身份是|秘密身份是|实际上是)", public_value)
                if leak and (cleaned := public_value[:leak.start()].rstrip(" ，,。；;")):
                    changes.append({"path": f"/characters/{index}/{field_name}", "value": cleaned + "。"})
        for field_name in ("relationships", "relationship_facets"):
            original = getattr(character, field_name)
            remapped = {}
            for target, value in original.items():
                resolved = target if target in names else aliases.get(target, target)
                if field_name == "relationship_facets":
                    facets = {}
                    for key, score in value.items():
                        dimension = key if key in dimensions else dimension_aliases.get(key, key)
                        # Numeric facets are optional summaries; never invent a
                        # new world dimension from an unknown model key.
                        if dimension in dimensions:
                            facets[dimension] = score
                    value = facets
                if resolved in remapped and resolved != target:
                    continue
                remapped[resolved] = value
            if remapped != original:
                changes.append({"path": f"/characters/{index}/{field_name}", "value": remapped})
    return (apply_scenario_patch(package, {"changes": changes}) if changes else package), changes


def recovery_advice(error):
    stalled = error.code in {"scenario_repair_stalled", "structured_output_stalled"}
    return {"resumable": not stalled,
            "recovery_action": "revise_or_restart" if stalled else "resume_after_check",
            "recovery_version": REPAIR_POLICY_VERSION}
