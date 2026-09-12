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
    enforce(prompt)
    return measured_call(
        state.model_requests, stage="simulation", purpose=purpose,
        model=getattr(llm, "model_name", "configured"),
        operation=lambda: llm.complete(prompt, max_tokens=max_tokens),
    )
