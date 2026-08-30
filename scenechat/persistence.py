from __future__ import annotations

import json
import sqlite3
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .knowledge import ExperimentKnowledgeBase, build_knowledge_documents
from .models import AbilityState, AgentState, MemoryRecord, Message, SimulationState
from .scenario import CharacterSpec, ScenarioBrief, ScenarioPackage, WorldSpec


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Small SQLite repository for user-owned simulation snapshots."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS story_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    payload TEXT NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_story_sessions_updated_at "
                "ON story_sessions(updated_at DESC)"
            )

    def save(self, payload: dict[str, Any]) -> None:
        session = payload["session"]
        scenario = payload["scenario"]
        simulation = payload["simulation"]
        now = utc_now()
        created_at = session.get("created_at") or now
        status = "ended" if session.get("ended") else simulation.get("run_status", "running")
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO story_sessions (
                    id, title, prompt, scene, created_at, updated_at,
                    status, turn_count, payload, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    prompt=excluded.prompt,
                    scene=excluded.scene,
                    updated_at=excluded.updated_at,
                    status=excluded.status,
                    turn_count=excluded.turn_count,
                    payload=excluded.payload,
                    schema_version=excluded.schema_version
                """,
                (
                    session["id"],
                    scenario.get("world", {}).get("title") or "未命名推演",
                    session.get("prompt", ""),
                    session.get("scene", ""),
                    created_at,
                    now,
                    status,
                    int(simulation.get("turn_count", 0)),
                    serialized,
                    SCHEMA_VERSION,
                ),
            )

    def load(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM story_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, prompt, scene, created_at, updated_at,
                       status, turn_count
                FROM story_sessions
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, session_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM story_sessions WHERE id = ?",
                (session_id,),
            )
        return cursor.rowcount > 0

    def clear(self) -> int:
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM story_sessions").fetchone()[0]
            connection.execute("DELETE FROM story_sessions")
        return int(count)


def scenario_from_dict(data: dict[str, Any]) -> ScenarioPackage:
    return ScenarioPackage(
        brief=ScenarioBrief.from_mapping(data.get("brief") or {}),
        world=WorldSpec.from_mapping(data.get("world") or {}),
        characters=[
            CharacterSpec.from_mapping(item, index)
            for index, item in enumerate(data.get("characters") or [], start=1)
            if isinstance(item, dict)
        ],
        warnings=[str(item) for item in data.get("warnings") or []],
    )


def _known_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


def agent_from_dict(data: dict[str, Any]) -> AgentState:
    values = _known_fields(AgentState, data)
    values["ability_states"] = {
        key: AbilityState(**_known_fields(AbilityState, value))
        for key, value in (data.get("ability_states") or {}).items()
        if isinstance(value, dict)
    }
    values["memories"] = [
        MemoryRecord(**_known_fields(MemoryRecord, value))
        for value in data.get("memories") or []
        if isinstance(value, dict)
    ]
    return AgentState(**values)


def state_from_dict(
    scene: str,
    package: ScenarioPackage,
    data: dict[str, Any],
) -> SimulationState:
    agents = [
        agent_from_dict(item)
        for item in data.get("agents") or []
        if isinstance(item, dict)
    ]
    state = SimulationState(scene, agents, world_spec=package.world)
    state.history = [
        Message(**_known_fields(Message, item))
        for item in data.get("history") or []
        if isinstance(item, dict)
    ]
    state.turn_count = int(data.get("turn_count", len(state.history)))
    state.agent_turn_count = int(data.get("agent_turn_count", 0))
    state.narration_count = int(data.get("narration_count", 0))
    state._scheduler_index = int(data.get("scheduler_index", 0))
    state.current_phase = str(data.get("current_phase") or state.current_phase)
    state.phase_action_log = set(data.get("phase_action_log") or [])
    state.world_state = dict(data.get("world_state") or {})
    state.public_rules = list(data.get("public_rules") or state.public_rules)
    state.termination_conditions = list(
        data.get("termination_conditions") or state.termination_conditions
    )
    state.ended = bool(data.get("ended", False))
    state.end_reason = str(data.get("end_reason") or "")
    state.end_kind = str(data.get("end_kind") or "")
    state.winner = str(data.get("winner") or "")
    state.run_status = str(data.get("run_status") or "running")
    state.failed_generation_count = int(data.get("failed_generation_count", 0))
    state.votes = dict(data.get("votes") or {})
    state.pending_events = list(data.get("pending_events") or [])
    state.protected_agents = set(data.get("protected_agents") or [])
    requested_order = [str(item) for item in data.get("agent_order") or []]
    state.agent_order = [name for name in requested_order if name in state.agents]
    if not state.agent_order:
        state.agent_order = list(state.agents)
    return state


def runtime_session_from_export(payload: dict[str, Any]) -> dict[str, Any]:
    session_data = payload["session"]
    package = scenario_from_dict(payload["scenario"])
    session_id = str(session_data["id"])
    scene = str(session_data.get("scene") or package.world.opening_scene)
    state = state_from_dict(scene, package, payload["simulation"])
    documents = build_knowledge_documents(
        package.public_worldview_markdown,
        package.characters_markdown,
        session_id.replace("-", ""),
        director_notes=package.world.director_notes_markdown,
        facts=package.world.facts,
    )
    page_requests = {
        str(item.get("request_id")): {
            "page": item.get("page"),
            "status": "retryable" if item.get("status") == "in_progress" else item.get("status"),
            "batch_size": item.get("batch_size"),
            "messages": list(item.get("messages") or []),
            "done": item.get("done"),
        }
        for item in payload.get("page_requests") or []
        if isinstance(item, dict) and item.get("request_id")
    }
    return {
        "created_at": session_data.get("created_at", ""),
        "prompt": str(session_data.get("prompt") or ""),
        "scene": scene,
        "worldview": package.public_worldview_markdown,
        "characters": package.public_characters_markdown,
        "public_characters": package.public_characters_markdown,
        "state": state,
        "knowledge_base": ExperimentKnowledgeBase(
            session_id.replace("-", ""), None, None, documents
        ),
        "scenario": package,
        "simulation_llm": None,
        "page": int(session_data.get("page", 0)),
        "ended": bool(session_data.get("ended", state.ended)),
        "page_requests": page_requests,
    }
