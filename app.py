import uuid
import json
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict

from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS

from scenechat.errors import SceneChatError, stage_error
from scenechat.generation import (
    generate_character_specs,
    generate_scenario_package,
    generate_scenario_brief,
    generate_world_spec,
    repair_scenario_package,
)
from scenechat.knowledge import build_experiment_knowledge_base, requires_vector_index
from scenechat.models import Intervention, Message, SimulationState
from scenechat.interventions import (
    has_blocking_conflicts,
    preview_intervention,
    public_intervention,
    validate_intervention,
)
from scenechat.pacing import PacingPolicy, initialize_arc
from scenechat.preflight import (
    validate_embedding_model_availability,
    validate_generation_model_availability,
)
from scenechat.persistence import SCHEMA_VERSION, SessionStore, runtime_session_from_export
from scenechat.providers import get_simulation_llm
from scenechat.scenario import (
    ScenarioPackage,
    ScenarioValidationError,
    agents_from_character_specs,
    normalize_scenario_phase_references,
    validate_scenario_package,
)
from scenechat.simulation import simulate_next_event
from scenechat.storage import save_experiment_documents

app = Flask(__name__)
cors_origins = [
    item.strip()
    for item in os.getenv(
        "SCENECHAT_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if item.strip()
]
CORS(app, resources={r"/api/*": {"origins": cors_origins}})

story_sessions: Dict[str, dict] = {}
session_store = SessionStore(os.getenv("SCENECHAT_DB_PATH", "data/scenechat.db"))

DEFAULT_BATCH_SIZE = 10
MAX_BATCH_SIZE = 10


def positive_int_env(name: str, default: int, *, maximum: int) -> int:
    """Read a bounded positive integer without making startup fragile."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        app.logger.warning("config.invalid name=%s value=%r fallback=%s", name, raw_value, default)
        return default
    if value < 1 or value > maximum:
        app.logger.warning("config.out_of_range name=%s value=%s fallback=%s", name, value, default)
        return default
    return value


MAX_TURNS = positive_int_env("SCENECHAT_MAX_TURNS", 120, maximum=1000)


def load_story_session(session_id: str) -> dict | None:
    session = story_sessions.get(session_id)
    if session is not None:
        return session
    try:
        payload = session_store.load(session_id)
        if payload is None:
            return None
        session = runtime_session_from_export(payload)
        session["stream_lock"] = threading.Lock()
        session["operation_lock"] = threading.Lock()
        story_sessions[session_id] = session
        return session
    except Exception:
        app.logger.exception("story.restore status=failed session=%s", session_id)
        return None


def persist_story_session(session_id: str, session: dict) -> None:
    if session.get("deleted"):
        return
    try:
        session_store.save(full_session_export(session_id, session))
    except Exception:
        app.logger.exception("story.persist status=failed session=%s", session_id)


def error_response(error: SceneChatError):
    return jsonify(error.to_payload()), error.status_code


def request_error(code: str, message: str, status_code: int = 400):
    return error_response(
        SceneChatError(
            code,
            message,
            stage="request",
            status_code=status_code,
        )
    )


def acquire_session_operation(session: dict, expected_revision):
    """Acquire a per-session mutation slot after optimistic revision validation."""

    try:
        expected = int(expected_revision) if expected_revision is not None else None
    except (TypeError, ValueError):
        return None, request_error("state_revision_invalid", "状态版本无效，请同步会话后重试。")
    state: SimulationState = session["state"]
    initialize_arc(state)
    stream_lock = session.setdefault("stream_lock", threading.Lock())
    operation_lock = session.setdefault("operation_lock", threading.Lock())
    with stream_lock:
        if operation_lock.locked():
            return None, request_error(
                "session_operation_in_progress",
                "当前会话正在执行另一项操作，请等待完成后重试。",
                status_code=409,
            )
        if expected is not None and expected != state.revision:
            return None, request_error(
                "state_revision_conflict",
                "剧情状态已经更新，请同步最新内容后再操作。",
                status_code=409,
            )
        if not operation_lock.acquire(blocking=False):
            return None, request_error(
                "session_operation_in_progress",
                "当前会话正在执行另一项操作，请稍后重试。",
                status_code=409,
            )
    return operation_lock, None


def run_stage(session_id: str, stage: str, label: str, operation):
    started_at = time.perf_counter()
    app.logger.info(
        "story.start stage=%s label=%s status=started session=%s",
        stage,
        label,
        session_id,
    )
    try:
        result = operation()
    except Exception as exc:
        error = stage_error(stage, exc)
        app.logger.exception(
            "story.start stage=%s label=%s status=failed session=%s code=%s",
            stage,
            label,
            session_id,
            error.code,
        )
        raise error from exc

    app.logger.info(
        "story.start stage=%s label=%s status=completed session=%s duration_ms=%d",
        stage,
        label,
        session_id,
        int((time.perf_counter() - started_at) * 1000),
    )
    return result


BUILD_STAGE_LABELS = {
    "generation_preflight": "检查生成模型",
    "brief": "理解用户设定",
    "world": "构建世界与规则",
    "characters": "生成角色档案",
    "validation": "校验一致性",
    "embedding_preflight": "检查向量模型",
    "runtime": "准备模拟环境",
    "storage": "保存实验档案",
}


def build_progress_event(stage: str, status: str, **details):
    return {
        "type": "build_progress",
        "stage": stage,
        "label": BUILD_STAGE_LABELS[stage],
        "status": status,
        **details,
    }


def public_scenario_payload(package: ScenarioPackage, state: SimulationState):
    public_constraints = [
        asdict(item)
        for item in package.brief.constraints
        if item.visibility == "public"
    ]
    protected_count = len(package.brief.constraints) - len(public_constraints)
    locked_constraints = [
        item for item in package.brief.constraints if item.locked and item.content
    ]
    covered_ids = set(package.world.covered_constraint_ids)
    for fact in package.world.facts:
        covered_ids.update(fact.covered_constraint_ids)
    for character in package.characters:
        covered_ids.update(character.covered_constraint_ids)

    return {
        "title": package.world.title,
        "brief": {
            **package.brief.public_dict(),
            "assumptions": list(package.brief.assumptions),
            "contradictions": list(package.brief.contradictions),
        },
        "constraint_summary": {
            "total": len(package.brief.constraints),
            "public": len(public_constraints),
            "protected": protected_count,
            "locked": len(locked_constraints),
            "covered": sum(1 for item in locked_constraints if item.id in covered_ids),
        },
        "public_rules": package.world.public_rules,
        "phases": package.world.phases,
        "locations": package.world.locations,
        "scheduler": package.world.scheduler,
        "initial_state": state.public_world_state(),
        "termination_conditions": package.world.termination_conditions,
        "warnings": package.warnings,
        "characters": [
            {
                "id": character.id,
                "name": character.name,
                "public_identity": character.public_identity,
                "public_traits": character.public_traits,
                "public_background": character.public_background,
                "initial_location": character.initial_location,
                "has_private_context": bool(
                    character.private_identity.strip()
                    or character.goals
                    or character.knowledge
                    or character.abilities
                ),
            }
            for character in package.characters
        ],
    }


def story_ready_payload(
    session_id: str,
    package: ScenarioPackage,
    state: SimulationState,
    scene: str,
):
    return {
        "session_id": session_id,
        "page": 0,
        "isEnd": False,
        "revision": state.revision,
        "max_turns": MAX_TURNS,
        "worldview": package.public_worldview_markdown,
        "characters": package.public_characters_markdown,
        "scene": scene,
        "scenario": public_scenario_payload(package, state),
        "messages": [],
    }


def build_story_events(user_prompt: str, scene_override: str, session_id: str):
    yield build_progress_event("generation_preflight", "started")
    run_stage(
        session_id,
        "preflight",
        BUILD_STAGE_LABELS["generation_preflight"],
        validate_generation_model_availability,
    )
    yield build_progress_event("generation_preflight", "completed")

    yield build_progress_event("brief", "started")
    brief = run_stage(
        session_id,
        "scenario_generation",
        BUILD_STAGE_LABELS["brief"],
        lambda: generate_scenario_brief(user_prompt, scene_override),
    )
    yield build_progress_event(
        "brief",
        "completed",
        input_mode=brief.input_mode,
        constraint_count=len(brief.constraints),
    )

    yield build_progress_event("world", "started")
    world = run_stage(
        session_id,
        "scenario_generation",
        BUILD_STAGE_LABELS["world"],
        lambda: generate_world_spec(user_prompt, brief),
    )
    yield build_progress_event("world", "completed", title=world.title)

    yield build_progress_event("characters", "started")
    characters = run_stage(
        session_id,
        "scenario_generation",
        BUILD_STAGE_LABELS["characters"],
        lambda: generate_character_specs(user_prompt, brief, world),
    )
    yield build_progress_event(
        "characters", "completed", character_count=len(characters)
    )

    yield build_progress_event("validation", "started")
    package = ScenarioPackage(brief=brief, world=world, characters=characters)
    normalize_scenario_phase_references(package)
    issues = validate_scenario_package(package)
    repaired = False
    if issues:
        repaired = True
        package = run_stage(
            session_id,
            "scenario_generation",
            "结构化场景局部修复",
            lambda: repair_scenario_package(user_prompt, package, issues),
        )
        normalize_scenario_phase_references(package)
        issues = validate_scenario_package(package)
    if issues:
        raise stage_error("scenario_generation", ScenarioValidationError(issues))
    yield build_progress_event(
        "validation",
        "completed",
        repaired=repaired,
        warning_count=len(package.warnings),
    )

    public_worldview = package.public_worldview_markdown
    characters_markdown = package.characters_markdown
    public_characters = package.public_characters_markdown
    initial_scene = scene_override or package.world.opening_scene

    needs_vector_index = requires_vector_index(
        public_worldview,
        characters_markdown,
        package.world.director_notes_markdown,
        package.world.facts,
    )
    yield build_progress_event(
        "embedding_preflight",
        "started" if needs_vector_index else "skipped",
        reason="长背景需要语义索引" if needs_vector_index else "当前设定可直接注入",
    )
    if needs_vector_index:
        run_stage(
            session_id,
            "preflight",
            BUILD_STAGE_LABELS["embedding_preflight"],
            validate_embedding_model_availability,
        )
        yield build_progress_event("embedding_preflight", "completed")

    yield build_progress_event("runtime", "started")
    agents = run_stage(
        session_id,
        "character_parsing",
        "角色状态准备",
        lambda: agents_from_character_specs(package.characters),
    )
    state = SimulationState(initial_scene, agents, world_spec=package.world)
    initialize_arc(state)
    knowledge_base = run_stage(
        session_id,
        "index",
        "知识库创建",
        lambda: build_experiment_knowledge_base(
            public_worldview,
            characters_markdown,
            session_id.replace("-", ""),
            director_notes=package.world.director_notes_markdown,
            facts=package.world.facts,
        ),
    )
    simulation_llm = run_stage(
        session_id,
        "simulation_client",
        "推演客户端准备",
        get_simulation_llm,
    )
    yield build_progress_event("runtime", "completed")

    yield build_progress_event("storage", "started")
    run_stage(
        session_id,
        "storage",
        BUILD_STAGE_LABELS["storage"],
        lambda: save_experiment_documents(
            session_id,
            package.worldview_markdown,
            characters_markdown,
            scenario_payload=package.to_dict(),
        ),
    )
    story_sessions[session_id] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": user_prompt,
        "scene": initial_scene,
        "worldview": public_worldview,
        "characters": public_characters,
        "public_characters": public_characters,
        "state": state,
        "knowledge_base": knowledge_base,
        "scenario": package,
        "simulation_llm": simulation_llm,
        "page": 0,
        "ended": False,
        "page_requests": {},
        "stream_lock": threading.Lock(),
        "operation_lock": threading.Lock(),
    }
    persist_story_session(session_id, story_sessions[session_id])
    yield build_progress_event("storage", "completed")
    app.logger.info("story.start status=completed session=%s", session_id)
    yield {
        "type": "story_ready",
        "data": story_ready_payload(session_id, package, state, initial_scene),
    }



def message_to_frontend(msg: Message):
    return {
        "id": msg.turn,
        "speaker": msg.speaker,
        "action": msg.action,
        "speech": msg.speech,
        "display_text": msg.speech,
        "turn": msg.turn,
        "kind": msg.kind,
        "visibility": msg.visibility,
        "state_updates": msg.state_updates,
        "end_signal": msg.end_signal,
        "end_reason": msg.end_reason,
        "event_id": msg.event_id,
        "visibility_scopes": msg.scopes,
        "location": msg.location,
        "intent": {
            key: value
            for key, value in msg.intent.items()
            if key not in {
                "private_reason",
                "proposed_patch",
                "relationship_updates",
                "short_term_state",
                "memory_candidates",
                "used_memory_ids",
                "claim_updates",
            }
        },
    }


def replay_page_response(page_request: dict, session_id: str):
    def replay():
        yield json.dumps({
            "type": "page_start",
            "page": page_request["page"],
            "session_id": session_id,
            "request_id": page_request["request_id"],
            "replayed": True,
        }, ensure_ascii=False) + "\n"
        for cached_message in page_request["messages"]:
            yield json.dumps({
                "type": "message",
                "page": page_request["page"],
                "request_id": page_request["request_id"],
                "message": cached_message,
                "replayed": True,
            }, ensure_ascii=False) + "\n"
        yield json.dumps(page_request["done"], ensure_ascii=False) + "\n"

    return Response(replay(), mimetype="application/x-ndjson")


@app.route("/api/story/start", methods=["POST"])
def start_story():
    data = request.get_json(silent=True)
    if not data:
        return request_error(
            "request_body_missing",
            "请求内容为空，请填写实验设定后重试。",
        )

    user_prompt = (data.get("prompt") or "").strip()
    scene_override = (data.get("scene") or "").strip()

    if not user_prompt:
        return request_error("prompt_missing", "请输入实验设定后再开始生成。")

    session_id = str(uuid.uuid4())
    app.logger.info("story.start status=received session=%s", session_id)

    try:
        run_stage(
            session_id,
            "preflight",
            BUILD_STAGE_LABELS["generation_preflight"],
            validate_generation_model_availability,
        )
        package = run_stage(
            session_id,
            "scenario_generation",
            "生成并校验结构化场景",
            lambda: generate_scenario_package(user_prompt, scene_override),
        )
        public_worldview = package.public_worldview_markdown
        characters_markdown = package.characters_markdown
        initial_scene = scene_override or package.world.opening_scene
        if requires_vector_index(
            public_worldview,
            characters_markdown,
            package.world.director_notes_markdown,
            package.world.facts,
        ):
            run_stage(
                session_id,
                "preflight",
                BUILD_STAGE_LABELS["embedding_preflight"],
                validate_embedding_model_availability,
            )
        agents = run_stage(
            session_id,
            "character_parsing",
            "角色状态准备",
            lambda: agents_from_character_specs(package.characters),
        )
        state = SimulationState(initial_scene, agents, world_spec=package.world)
        initialize_arc(state)
        knowledge_base = run_stage(
            session_id,
            "index",
            "知识库创建",
            lambda: build_experiment_knowledge_base(
                public_worldview,
                characters_markdown,
                session_id.replace("-", ""),
                director_notes=package.world.director_notes_markdown,
                facts=package.world.facts,
            ),
        )
        simulation_llm = run_stage(
            session_id,
            "simulation_client",
            "推演客户端准备",
            get_simulation_llm,
        )
        run_stage(
            session_id,
            "storage",
            BUILD_STAGE_LABELS["storage"],
            lambda: save_experiment_documents(
                session_id,
                package.worldview_markdown,
                characters_markdown,
                scenario_payload=package.to_dict(),
            ),
        )
        public_characters = package.public_characters_markdown
        story_sessions[session_id] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt": user_prompt,
            "scene": initial_scene,
            "worldview": public_worldview,
            "characters": public_characters,
            "public_characters": public_characters,
            "state": state,
            "knowledge_base": knowledge_base,
            "scenario": package,
            "simulation_llm": simulation_llm,
            "page": 0,
            "ended": False,
            "page_requests": {},
            "stream_lock": threading.Lock(),
            "operation_lock": threading.Lock(),
        }
        persist_story_session(session_id, story_sessions[session_id])
        return jsonify(story_ready_payload(session_id, package, state, initial_scene))

    except SceneChatError as error:
        return error_response(error)
    except Exception as exc:
        error = stage_error("internal", exc)
        app.logger.exception(
            "story.start status=failed session=%s code=%s",
            session_id,
            error.code,
        )
        return error_response(error)


@app.route("/api/story/start-stream", methods=["POST"])
def start_story_stream():
    data = request.get_json(silent=True)
    if not data:
        return request_error(
            "request_body_missing",
            "请求内容为空，请填写实验设定后重试。",
        )

    user_prompt = (data.get("prompt") or "").strip()
    scene_override = (data.get("scene") or "").strip()
    if not user_prompt:
        return request_error("prompt_missing", "请输入实验设定后再开始生成。")

    session_id = str(uuid.uuid4())
    app.logger.info("story.start status=received session=%s", session_id)

    def generate():
        try:
            for event in build_story_events(user_prompt, scene_override, session_id):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except SceneChatError as error:
            yield json.dumps(
                {
                    "type": "error",
                    "error": error.to_payload()["error"],
                },
                ensure_ascii=False,
            ) + "\n"
        except Exception as exc:
            error = stage_error("internal", exc)
            app.logger.exception(
                "story.start status=failed session=%s code=%s",
                session_id,
                error.code,
            )
            yield json.dumps(
                {
                    "type": "error",
                    "error": error.to_payload()["error"],
                },
                ensure_ascii=False,
            ) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
    )


@app.route("/api/story/next-stream", methods=["POST"])
def next_story_page_stream():
    data = request.get_json(silent=True)
    if not data:
        return request_error(
            "request_body_missing",
            "请求内容为空，请重新操作。",
        )

    session_id = data.get("session_id")
    if not session_id:
        return request_error(
            "session_id_missing",
            "故事会话信息缺失，请返回首页重新开始。",
        )

    session = load_story_session(session_id)
    if not session:
        return request_error(
            "session_not_found",
            "故事会话不存在或已经失效，请返回首页重新开始。",
            status_code=404,
        )

    if session["ended"]:
        completed_request = session.get("page_requests", {}).get(
            str(data.get("request_id") or "").strip()
        )
        if completed_request and completed_request.get("status") == "completed":
            return replay_page_response(completed_request, session_id)
        return jsonify({
            "session_id": session_id,
            "page": session["page"],
            "isEnd": True,
            "messages": [],
        }), 200

    request_id = str(data.get("request_id") or f"page-{session['page'] + 1}").strip()[:120]
    if not request_id:
        return request_error("request_id_missing", "页面请求标识缺失，请重试当前页面。")
    expected_page = data.get("expected_page")
    try:
        expected_page = int(expected_page) if expected_page is not None else session["page"] + 1
    except (TypeError, ValueError):
        return request_error("page_number_invalid", "页面序号无效，请同步会话后重试。")
    expected_revision = data.get("expected_revision")
    try:
        expected_revision = (
            int(expected_revision) if expected_revision is not None else None
        )
    except (TypeError, ValueError):
        return request_error("state_revision_invalid", "状态版本无效，请同步会话后重试。")

    batch_size = data.get("batch_size", DEFAULT_BATCH_SIZE)
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError):
        batch_size = DEFAULT_BATCH_SIZE
    batch_size = max(1, min(batch_size, MAX_BATCH_SIZE))

    state: SimulationState = session["state"]
    knowledge_base = session["knowledge_base"]
    simulation_llm = session.get("simulation_llm")
    page_requests = session.setdefault("page_requests", {})
    stream_lock = session.setdefault("stream_lock", threading.Lock())
    operation_lock = session.setdefault("operation_lock", threading.Lock())
    with stream_lock:
        page_request = page_requests.get(request_id)
        if page_request is not None and page_request["status"] == "in_progress":
            return request_error(
                "page_request_in_progress",
                "当前页面仍在生成，请稍候；不要重复推进。",
                status_code=409,
            )
        if page_request is not None and page_request["status"] == "completed":
            return replay_page_response(page_request, session_id)
        if operation_lock.locked():
            return request_error(
                "session_operation_in_progress",
                "当前会话正在执行另一项操作，请等待本幕完成后重试。",
                status_code=409,
            )
        if page_request is None:
            if expected_revision is not None and expected_revision != getattr(state, "revision", 0):
                return request_error(
                    "state_revision_conflict",
                    "剧情状态已经更新，请同步最新内容后再继续。",
                    status_code=409,
                )
            if expected_page != session["page"] + 1:
                return request_error(
                    "page_state_conflict",
                    "前端页码与后端状态不一致，请同步会话后重试。",
                    status_code=409,
                )
            page_request = {
                "request_id": request_id,
                "page": expected_page,
                "status": "in_progress",
                "messages": [],
                "batch_size": batch_size,
                "done": None,
            }
            page_requests[request_id] = page_request
        else:
            # A retryable request resumes after its already committed messages.
            page_request["status"] = "in_progress"
        if not operation_lock.acquire(blocking=False):
            page_request["status"] = "retryable"
            return request_error(
                "session_operation_in_progress",
                "当前会话正在执行另一项操作，请稍后重试。",
                status_code=409,
            )

    next_page_number = page_request["page"]

    def generate():
        sent_count = 0

        yield json.dumps({
            "type": "page_start",
            "page": next_page_number,
            "session_id": session_id,
            "request_id": request_id,
            "resumed_count": len(page_request["messages"]),
            "revision": getattr(state, "revision", 0),
        }, ensure_ascii=False) + "\n"

        for cached_message in page_request["messages"]:
            yield json.dumps({
                "type": "message",
                "page": next_page_number,
                "request_id": request_id,
                "message": cached_message,
                "replayed": True,
            }, ensure_ascii=False) + "\n"

        remaining_turns = max(0, MAX_TURNS - state.turn_count)
        target_count = min(page_request["batch_size"], len(page_request["messages"]) + remaining_turns)

        try:
            while len(page_request["messages"]) < target_count:
                if not getattr(state, "can_continue", not state.ended):
                    break
                msg = simulate_next_event(state, knowledge_base, llm=simulation_llm)
                if msg is None:
                    if not getattr(state, "can_continue", True):
                        break
                    continue

                state.add_message(msg)
                sent_count += 1
                serialized = message_to_frontend(msg)
                page_request["messages"].append(serialized)

                yield json.dumps({
                    "type": "message",
                    "page": next_page_number,
                    "request_id": request_id,
                    "message": serialized,
                }, ensure_ascii=False) + "\n"
        except GeneratorExit:
            page_request["status"] = "retryable"
            raise
        except Exception as exc:
            page_request["status"] = "retryable"
            error = stage_error("simulation", exc)
            app.logger.exception(
                "story.stream status=failed session=%s page=%s code=%s",
                session_id,
                next_page_number,
                error.code,
            )
            yield json.dumps({
                "type": "error",
                "page": next_page_number,
                "request_id": request_id,
                "retryable": True,
                "committed_count": len(page_request["messages"]),
                "error": error.to_payload()["error"],
            }, ensure_ascii=False) + "\n"
            return

        session["page"] = next_page_number

        is_blocked = getattr(state, "run_status", "running") == "blocked"
        is_end = state.ended or is_blocked or state.turn_count >= MAX_TURNS
        session["ended"] = is_end
        end_kind = state.end_kind if (state.ended or is_blocked) else (
            "safety_cap" if state.turn_count >= MAX_TURNS else ""
        )
        done_event = {
            "type": "page_done",
            "page": next_page_number,
            "session_id": session_id,
            "request_id": request_id,
            "isEnd": is_end,
            "count": sent_count,
            "total_count": len(page_request["messages"]),
            "run_status": getattr(state, "run_status", "running"),
            "end_kind": end_kind,
            "end_reason": state.end_reason if state.ended else (
                getattr(state, "end_reason", "") if is_blocked else (
                    "已达到安全轮次上限。" if state.turn_count >= MAX_TURNS else ""
                )
            ),
            "revision": getattr(state, "revision", 0),
        }
        page_request["done"] = done_event
        page_request["status"] = "completed"
        persist_story_session(session_id, session)
        yield json.dumps(done_event, ensure_ascii=False) + "\n"

    def locked_generate():
        try:
            yield from generate()
        finally:
            if operation_lock.locked():
                operation_lock.release()

    return Response(
        stream_with_context(locked_generate()),
        mimetype="application/x-ndjson"
    )


@app.route("/api/story/session/<session_id>/interventions/preview", methods=["POST"])
def preview_story_intervention(session_id):
    session = load_story_session(session_id)
    if not session:
        return request_error("session_not_found", "故事会话不存在或已经失效。", status_code=404)
    data = request.get_json(silent=True) or {}
    raw_text = str(data.get("text") or "").strip()
    scope = str(data.get("scope") or "next_scene")
    if scope not in {"next_scene", "turns", "persistent"}:
        return request_error("intervention_scope_invalid", "干预作用范围无效。")
    try:
        duration = int(data.get("expires_after_turns", 6)) if scope == "turns" else None
    except (TypeError, ValueError):
        return request_error("intervention_duration_invalid", "持续轮数必须是整数。")
    if duration is not None and not 1 <= duration <= 100:
        return request_error("intervention_duration_invalid", "持续轮数必须在 1 到 100 之间。")
    operation_lock, error = acquire_session_operation(session, data.get("expected_revision"))
    if error:
        return error
    try:
        intervention = preview_intervention(
            session["state"], raw_text, scope=scope,
            expires_after_turns=duration, llm=session.get("simulation_llm"),
        )
        return jsonify({
            "intervention": public_intervention(intervention),
            "revision": session["state"].revision,
        })
    except ValueError as exc:
        return request_error("intervention_preview_invalid", str(exc))
    except Exception as exc:
        app.logger.exception("story.intervention_preview status=failed session=%s", session_id)
        return error_response(stage_error("intervention", exc))
    finally:
        operation_lock.release()


@app.route("/api/story/session/<session_id>/interventions", methods=["POST"])
def confirm_story_intervention(session_id):
    session = load_story_session(session_id)
    if not session:
        return request_error("session_not_found", "故事会话不存在或已经失效。", status_code=404)
    data = request.get_json(silent=True) or {}
    source = data.get("intervention")
    if not isinstance(source, dict):
        return request_error("intervention_missing", "请先预检干预内容。")
    operation_lock, error = acquire_session_operation(session, data.get("expected_revision"))
    if error:
        return error
    try:
        state: SimulationState = session["state"]
        intervention = validate_intervention(state, Intervention.from_mapping(source))
        if not intervention.raw_text or not intervention.normalized_directive:
            return request_error("intervention_invalid", "干预预检结果不完整，请重新预检。")
        if any(item.id == intervention.id for item in state.interventions):
            return request_error("intervention_duplicate", "这条干预已经提交。", status_code=409)
        blocking = has_blocking_conflicts(intervention)
        force = bool(data.get("force", False))
        if blocking and intervention.mode != "override":
            return request_error(
                "intervention_requires_override",
                "这项改动与现有剧情冲突；请修改指令，或明确要求强制改写。",
                status_code=409,
            )
        if intervention.mode == "override" and not force:
            return request_error(
                "intervention_confirmation_required", "强制改写需要再次明确确认。", status_code=409,
            )
        intervention.status = "pending"
        intervention.created_at_turn = state.turn_count
        intervention.effective_after_turn = state.turn_count
        state.register_intervention(intervention)
        persist_story_session(session_id, session)
        return jsonify({
            "intervention": public_intervention(intervention), "revision": state.revision,
        }), 201
    finally:
        operation_lock.release()


@app.route("/api/story/session/<session_id>/interventions/<intervention_id>", methods=["DELETE"])
def cancel_story_intervention(session_id, intervention_id):
    session = load_story_session(session_id)
    if not session:
        return request_error("session_not_found", "故事会话不存在或已经失效。", status_code=404)
    data = request.get_json(silent=True) or {}
    operation_lock, error = acquire_session_operation(session, data.get("expected_revision"))
    if error:
        return error
    try:
        state: SimulationState = session["state"]
        intervention = next((item for item in state.interventions if item.id == intervention_id), None)
        if intervention is None:
            return request_error("intervention_not_found", "没有找到这条干预。", status_code=404)
        if intervention.status in {"cancelled", "rejected"}:
            return jsonify({"intervention": public_intervention(intervention), "revision": state.revision})
        if intervention.status == "applied" and intervention.mode in {"event", "override"}:
            return request_error(
                "intervention_already_applied",
                "该事件已经进入时间线，不能直接撤销；可提交一条新的修正干预。",
                status_code=409,
            )
        intervention.status = "cancelled"
        state.bump_revision()
        persist_story_session(session_id, session)
        return jsonify({"intervention": public_intervention(intervention), "revision": state.revision})
    finally:
        operation_lock.release()


@app.route("/api/story/session/<session_id>/pace", methods=["PATCH"])
def update_story_pace(session_id):
    session = load_story_session(session_id)
    if not session:
        return request_error("session_not_found", "故事会话不存在或已经失效。", status_code=404)
    data = request.get_json(silent=True) or {}
    try:
        pace = int(data.get("pace"))
    except (TypeError, ValueError):
        return request_error("pace_invalid", "剧情推进速度必须是 0 到 100 的整数。")
    if not 0 <= pace <= 100:
        return request_error("pace_invalid", "剧情推进速度必须在 0 到 100 之间。")
    operation_lock, error = acquire_session_operation(session, data.get("expected_revision"))
    if error:
        return error
    try:
        state: SimulationState = session["state"]
        state.set_pace(pace)
        persist_story_session(session_id, session)
        policy = PacingPolicy.from_value(pace)
        return jsonify({
            "arc_state": asdict(state.arc_state),
            "pace_label": policy.label,
            "revision": state.revision,
        })
    finally:
        operation_lock.release()


@app.route("/api/story/session/<session_id>", methods=["GET"])
def get_story_session(session_id):
    session = load_story_session(session_id)
    if not session:
        return request_error(
            "session_not_found",
            "故事会话不存在或已经失效，请返回首页重新开始。",
            status_code=404,
        )

    state: SimulationState = session["state"]
    initialize_arc(state)

    pages = []
    completed_requests = [
        item
        for item in session.get("page_requests", {}).values()
        if item.get("status") == "completed"
    ]
    for page_request in sorted(completed_requests, key=lambda item: item.get("page") or 0):
        done = page_request.get("done") or {}
        pages.append({
            "page": page_request.get("page"),
            "messages": list(page_request.get("messages") or []),
            "isEnd": bool(done.get("isEnd", False)),
            "endReason": done.get("end_reason", ""),
            "endKind": done.get("end_kind", ""),
            "runStatus": done.get("run_status", "running"),
            "requestId": done.get("request_id"),
        })

    return jsonify({
        "session_id": session_id,
        "page": session["page"],
        "isEnd": session["ended"],
        "prompt": session["prompt"],
        "scene": session["scene"],
        "worldview": session["scenario"].public_worldview_markdown,
        "characters": session["public_characters"],
        "scenario": public_scenario_payload(session["scenario"], state),
        "max_turns": MAX_TURNS,
        "revision": state.revision,
        "arc_state": {
            "pace": state.arc_state.pace,
            "progress": state.arc_state.progress,
            "tension": state.arc_state.tension,
            "target_end_turn": state.arc_state.target_end_turn,
            "turns_since_progress": state.arc_state.turns_since_progress,
            "active_beat_ids": state.arc_state.active_beat_ids,
            "resolved_beat_ids": state.arc_state.resolved_beat_ids,
        },
        "interventions": [public_intervention(item) for item in state.interventions],
        "turn_count": state.turn_count,
        "current_phase": state.current_phase,
        "world_state": state.public_world_state(),
        "end_reason": state.end_reason,
        "end_kind": state.end_kind,
        "run_status": getattr(state, "run_status", "running"),
        "winner": getattr(state, "winner", ""),
        "public_state": state.public_state_summary(),
        "pages": pages,
        "agents": [
            {
                "id": agent.id,
                "name": agent.name,
                "public_profile": agent.public_profile,
                "active": agent.active,
                "alive": agent.alive,
                "current_location": agent.current_location,
            }
            for agent in state.agents.values()
        ],
        "history": [message_to_frontend(msg) for msg in state.history],
    })


def full_session_export(session_id: str, session: dict) -> dict:
    """Build an explicit owner export, including private scenario and agent state."""
    state: SimulationState = session["state"]
    initialize_arc(state)
    package: ScenarioPackage = session["scenario"]
    page_requests = []
    for request_id, page_request in session.get("page_requests", {}).items():
        page_requests.append({
            "request_id": request_id,
            "page": page_request.get("page"),
            "status": page_request.get("status"),
            "batch_size": page_request.get("batch_size"),
            "messages": list(page_request.get("messages", [])),
            "done": page_request.get("done"),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "session": {
            "id": session_id,
            "created_at": session.get("created_at", ""),
            "prompt": session["prompt"],
            "scene": session["scene"],
            "page": session["page"],
            "ended": session["ended"],
            "max_turns": MAX_TURNS,
        },
        "scenario": package.to_dict(),
        "documents": {
            "worldview_markdown": package.worldview_markdown,
            "characters_markdown": package.characters_markdown,
            "public_worldview_markdown": package.public_worldview_markdown,
            "public_characters_markdown": package.public_characters_markdown,
        },
        "simulation": {
            "revision": state.revision,
            "turn_count": state.turn_count,
            "agent_turn_count": state.agent_turn_count,
            "narration_count": state.narration_count,
            "scheduler_index": state._scheduler_index,
            "current_phase": state.current_phase,
            "phase_action_log": sorted(state.phase_action_log),
            "world_state": state.world_state,
            "public_rules": state.public_rules,
            "termination_conditions": state.termination_conditions,
            "ended": state.ended,
            "end_reason": state.end_reason,
            "end_kind": state.end_kind,
            "winner": state.winner,
            "run_status": state.run_status,
            "failed_generation_count": state.failed_generation_count,
            "dialogue_quality_retry_count": state.dialogue_quality_retry_count,
            "dialogue_quality_issue_counts": dict(state.dialogue_quality_issue_counts),
            "conversation_threads": [
                asdict(item) for item in state.conversation_threads.values()
            ],
            "last_scheduler_decision": dict(state.last_scheduler_decision),
            "votes": state.votes,
            "pending_events": state.pending_events,
            "interventions": [asdict(item) for item in state.interventions],
            "arc_state": asdict(state.arc_state),
            "protected_agents": sorted(state.protected_agents),
            "agent_order": state.agent_order,
            "agents": [asdict(agent) for agent in state.agents.values()],
            "history": [asdict(message) for message in state.history],
        },
        "page_requests": page_requests,
    }


@app.route("/api/story/session/<session_id>/export", methods=["GET"])
def export_story_session(session_id):
    session = load_story_session(session_id)
    if not session:
        return request_error(
            "session_not_found",
            "故事会话不存在或已经失效，无法导出。",
            status_code=404,
        )

    content = json.dumps(
        full_session_export(session_id, session),
        ensure_ascii=False,
        indent=2,
    )
    return Response(
        content,
        mimetype="application/json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="scenechat-{session_id}.json"'
            )
        },
    )


@app.route("/api/story/session/<session_id>", methods=["DELETE"])
def delete_story_session(session_id):
    """Delete one user-owned simulation from memory and SQLite."""
    session = story_sessions.pop(session_id, None)
    if session is not None:
        session["deleted"] = True
    deleted = session_store.delete(session_id) or session is not None
    return jsonify({"session_id": session_id, "deleted": deleted})


@app.route("/api/story/sessions", methods=["GET"])
def list_story_sessions():
    return jsonify({"sessions": session_store.list()})


@app.route("/api/story/sessions", methods=["DELETE"])
def clear_story_sessions():
    for session in story_sessions.values():
        session["deleted"] = True
    story_sessions.clear()
    deleted_count = session_store.clear()
    return jsonify({"deleted_count": deleted_count})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
