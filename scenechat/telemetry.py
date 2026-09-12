"""Request metadata only: never store prompts, keys, or provider error bodies."""
import logging
import time
from datetime import datetime, timezone


def measured_call(records, *, stage, purpose, model, operation):
    started = time.monotonic()
    response = None
    error = None
    try:
        response = operation()
        return response
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        raw = getattr(response, "raw", response)
        usage = getattr(raw, "usage", None)
        def count(key):
            value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
            return value if isinstance(value, int) and value >= 0 else None
        record = {
            "at": datetime.now(timezone.utc).isoformat(), "stage": stage,
            "purpose": purpose, "model": str(model),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "prompt_tokens": count("prompt_tokens"),
            "completion_tokens": count("completion_tokens"),
            "total_tokens": count("total_tokens"), "error_type": error,
        }
        records.append(record)
        logging.getLogger(__name__).info("model.request %s", record)


def complete(state, llm, prompt, *, purpose, max_tokens):
    from .context_budget import enforce
    from .build_control import CURRENT_BUILD
    from .config import config_int
    from .recovery import bounded_operation, transport_call
    if CURRENT_BUILD.get() is None:
        return bounded_operation("simulation")(complete)(state, llm, prompt, purpose=purpose, max_tokens=max_tokens)
    enforce(prompt)
    return transport_call(
        lambda: measured_call(
            state.model_requests, stage=CURRENT_BUILD.get().stage, purpose=purpose,
            model=getattr(llm, "model_name", "configured"),
            operation=lambda: llm.complete(prompt, max_tokens=max_tokens)),
        control=CURRENT_BUILD.get(),
        retries=0 if purpose == "beat_verification" else config_int(
            "simulation", "transport_retries", 1, minimum=0, maximum=2),
        limit=config_int("simulation", "max_requests_per_operation", 5, minimum=1, maximum=10),
    )
