"""Per-build cancellation, wall-clock budgets and safe progress metadata."""

import asyncio
import time
import threading
import logging
from contextlib import contextmanager
from contextvars import ContextVar

from .config import config_int
from .errors import SceneChatError

CURRENT_BUILD = ContextVar("scenechat_build", default=None)


class BuildControl:
    def __init__(self, emit=None):
        self.started = time.monotonic()
        self.deadline = self.started + config_int("scenario", "build_timeout_seconds", 600, minimum=10, maximum=3600)
        self.step_deadline = self.deadline
        self.stage = "preflight"
        self.cancelled = threading.Event()
        self.emit = emit or (lambda event: None)
        self.details = {}
        self.step_requests = 0
        self.guard = None
        self.model_requests = []

    def check(self):
        if self.guard:
            self.guard()
        if self.cancelled.is_set():
            raise SceneChatError("build_cancelled", "构建已取消，已完成的检查点已保留。", stage=self.stage, status_code=409)
        if time.monotonic() >= min(self.deadline, self.step_deadline):
            raise SceneChatError("build_deadline_exceeded", "构建已达到时间预算，已完成的检查点已保留，可从断点继续。", stage=self.stage, status_code=504)

    def begin_step(self, stage):
        self.stage = stage
        self.step_deadline = min(self.deadline, time.monotonic() + config_int(
            "scenario", "step_timeout_seconds", 240, minimum=5, maximum=1800
        ))
        self.details = {}
        self.step_requests = 0
        self.check()

    def progress(self, **details):
        self.details.update(details)
        logging.getLogger(__name__).info("story.build stage=%s elapsed=%.1f metadata=%s",
                                       self.stage, time.monotonic() - self.started,
                                       {key: value for key, value in details.items() if key != "issues"})
        self.emit(self.snapshot())

    def snapshot(self):
        now = time.monotonic()
        return {"type": "build_activity", "stage": self.stage,
                "elapsed_seconds": round(now - self.started, 1),
                "remaining_seconds": max(0, round(min(self.deadline, self.step_deadline) - now, 1)),
                **self.details}

    @contextmanager
    def activate(self):
        token = CURRENT_BUILD.set(self)
        try:
            self.check()
            yield self
        finally:
            CURRENT_BUILD.reset(token)


async def await_controlled(awaitable, control, timeout):
    """Cancel the actual async HTTP task, not just a waiting wrapper thread."""
    task = asyncio.ensure_future(awaitable)
    deadline = time.monotonic() + timeout
    try:
        while True:
            control.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("model request timed out")
            done, _ = await asyncio.wait({task}, timeout=min(0.1, max(0, deadline - time.monotonic())))
            if done:
                control.check()
                return task.result()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
