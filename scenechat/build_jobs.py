"""Resumable build stream. Private checkpoints never enter progress events."""

import queue
import threading
import time
import uuid

from .build_control import BuildControl
from .errors import SceneChatError, stage_error
from .recovery import recovery_advice


class BuildJobs:
    def __init__(self, store_getter, builder):
        self.store_getter = store_getter
        self.builder = builder
        self.active = {}
        self.lock = threading.Lock()

    def start(self, prompt, scene, build_id=None):
        store = self.store_getter()
        previous = store.load_build(build_id) if build_id else None
        if build_id and previous is None:
            raise SceneChatError("build_not_found", "构建检查点不存在，请重新开始。", stage="request", status_code=404)
        if (previous and previous["status"] == "failed"
                and previous["payload"].get("last_error", {}).get("resumable") is False
                and previous["payload"].get("prompt") == prompt and previous["payload"].get("scene") == scene):
            # Check before preflight: a stalled repair is not a transient outage.
            def stopped():
                yield {"type": "error", "build_id": build_id, **previous["payload"]["last_error"]}
            return stopped()
        build_id = build_id or str(uuid.uuid4())
        owner = str(uuid.uuid4())
        events = queue.Queue()
        control = BuildControl(events.put)
        try:
            payload = store.claim_build(build_id, owner, time.time() + 15, prompt, scene)
        except ValueError as exc:
            code = str(exc)
            message = "该构建仍在运行或取消中，请稍后继续。" if code == "build_in_progress" else "设定已改变，请开始新的构建。"
            raise SceneChatError(code, message, stage="request", status_code=409) from exc
        control.checkpoint = payload["checkpoint"]
        control.model_requests = payload.setdefault("model_requests", [])
        control.recovery = payload.setdefault("recovery", {})
        control.owner = owner
        last_guard = [0.0]

        def guard():
            if time.monotonic() - last_guard[0] < 1:
                return
            last_guard[0] = time.monotonic()
            row = store.load_build(build_id)
            if row is None or row["owner"] != owner or row["status"] == "cancelling":
                control.cancelled.set()

        control.guard = guard

        def save_checkpoint(key, value):
            control.check()
            payload["checkpoint"][key] = value
            if not store.update_build(build_id, owner, payload):
                control.cancelled.set()
                control.check()

        control.save_checkpoint = save_checkpoint
        def save_recovery():
            control.check()
            if not store.update_build(build_id, owner, payload):
                control.cancelled.set()
                control.check()
        control.save_recovery = save_recovery
        with self.lock:
            self.active[build_id] = control

        def work():
            try:
                with control.activate():
                    for event in self.builder(prompt, scene, build_id):
                        control.check()
                        events.put(event)
                    control.check()
                    store.update_build(build_id, owner, payload, "completed")
            except Exception as exc:
                error = stage_error("scenario_generation", exc)
                payload["last_error"] = {**error.to_payload(), **recovery_advice(error)}
                status = "cancelled" if control.cancelled.is_set() else "failed"
                store.update_build(build_id, owner, payload, status)
                events.put({"type": "error", "build_id": build_id,
                            **recovery_advice(error), "error": error.to_payload()["error"]})
            finally:
                with self.lock:
                    if self.active.get(build_id) is control:
                        self.active.pop(build_id, None)
                events.put(None)

        def stream():
            worker = threading.Thread(target=work, daemon=True, name=f"build-{build_id}")
            worker.start()
            last_heartbeat = time.monotonic()
            try:
                yield {"type": "build_started", "build_id": build_id,
                       "resumed": bool(control.checkpoint), "completed_stages": list(control.checkpoint)}
                while True:
                    row = store.load_build(build_id)
                    if row is None or row["owner"] != owner or row["status"] == "cancelling":
                        control.cancelled.set()
                    control.check()
                    try:
                        event = events.get(timeout=0.25)
                    except queue.Empty:
                        if time.monotonic() - last_heartbeat >= 1:
                            store.renew_build(build_id, owner, time.time() + 15)
                            yield control.snapshot()
                            last_heartbeat = time.monotonic()
                        continue
                    if event is None:
                        return
                    yield event
            except SceneChatError as exc:
                control.cancelled.set()
                store.update_build(build_id, owner, payload, "failed")
                yield {"type": "error", "build_id": build_id, "resumable": True,
                       "error": exc.to_payload()["error"]}
            finally:
                if worker.is_alive():
                    control.cancelled.set()
                    store.cancel_build(build_id)
                    # Give cooperative HTTP cancellation and checkpoint cleanup
                    # a short grace period; never wait for a full model timeout.
                    worker.join(timeout=1)

        return stream()

    def cancel(self, build_id):
        self.store_getter().cancel_build(build_id)
        with self.lock:
            control = self.active.get(build_id)
            if control:
                control.cancelled.set()

    def cancel_all(self):
        with self.lock:
            for control in self.active.values():
                control.cancelled.set()
