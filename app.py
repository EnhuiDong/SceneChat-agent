import uuid
import json
import threading
import time
from typing import Dict

from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS

from scenechat.errors import SceneChatError, stage_error
from scenechat.generation import generate_scenario_package
from scenechat.knowledge import build_experiment_knowledge_base, requires_vector_index
from scenechat.models import Message, SimulationState
from scenechat.preflight import (
    validate_embedding_model_availability,
    validate_generation_model_availability,
)
from scenechat.providers import get_simulation_llm
from scenechat.scenario import agents_from_character_specs
from scenechat.simulation import simulate_next_event
from scenechat.storage import save_experiment_documents

app = Flask(__name__)
CORS(app)

story_sessions: Dict[str, dict] = {}

DEFAULT_BATCH_SIZE = 10
MAX_BATCH_SIZE = 10
MAX_TURNS = 30


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
            if key not in {"private_reason", "proposed_patch", "relationship_updates"}
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
            "模型预检",
            validate_generation_model_availability,
        )
        package = run_stage(
            session_id,
            "scenario_generation",
            "结构化场景生成与校验",
            lambda: generate_scenario_package(user_prompt, scene_override),
        )
        worldview = package.worldview_markdown
        public_worldview = package.public_worldview_markdown
        characters = package.characters_markdown
        public_characters = package.public_characters_markdown
        initial_scene = scene_override or package.world.opening_scene
        if requires_vector_index(
            public_worldview,
            characters,
            package.world.director_notes_markdown,
            package.world.facts,
        ):
            run_stage(
                session_id,
                "preflight",
                "向量模型预检",
                validate_embedding_model_availability,
            )
        agents = run_stage(
            session_id,
            "character_parsing",
            "角色解析",
            lambda: agents_from_character_specs(package.characters),
        )
        state = SimulationState(initial_scene, agents, world_spec=package.world)
        knowledge_base = run_stage(
            session_id,
            "index",
            "知识库创建",
            lambda: build_experiment_knowledge_base(
                public_worldview,
                characters,
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
            "实验归档",
            lambda: save_experiment_documents(
                session_id,
                worldview,
                characters,
                scenario_payload=package.to_dict(),
            ),
        )

        story_sessions[session_id] = {
            "prompt": user_prompt,
            "scene": initial_scene,
            "worldview": public_worldview,
            "characters": public_characters,
            "public_characters": public_characters,
            "state": state,
            "knowledge_base": knowledge_base,
            "scenario": package,
            "simulation_llm": simulation_llm,
            "page": 0,         # 注意：现在从 0 开始，还没正式生成第一页
            "ended": False,
            "page_requests": {},
            "stream_lock": threading.Lock(),
        }

        app.logger.info("story.start status=completed session=%s", session_id)
        return jsonify({
            "session_id": session_id,
            "page": 0,
            "isEnd": False,
            "worldview": public_worldview,
            "characters": public_characters,
            "scene": initial_scene,
            "scenario": {
                "title": package.world.title,
                "brief": package.brief.public_dict(),
                "public_rules": package.world.public_rules,
                "phases": package.world.phases,
                "initial_state": state.public_world_state(),
                "termination_conditions": package.world.termination_conditions,
                "warnings": package.warnings,
            },
            "messages": [],
        })

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

    session = story_sessions.get(session_id)
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
    with stream_lock:
        page_request = page_requests.get(request_id)
        if page_request is not None and page_request["status"] == "in_progress":
            return request_error(
                "page_request_in_progress",
                "当前页面仍在生成，请稍候；不要重复推进。",
                status_code=409,
            )
        if page_request is None:
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
        elif page_request["status"] == "completed":
            return replay_page_response(page_request, session_id)
        else:
            # A retryable request resumes after its already committed messages.
            page_request["status"] = "in_progress"

    next_page_number = page_request["page"]

    def generate():
        sent_count = 0

        yield json.dumps({
            "type": "page_start",
            "page": next_page_number,
            "session_id": session_id,
            "request_id": request_id,
            "resumed_count": len(page_request["messages"]),
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
        }
        page_request["done"] = done_event
        page_request["status"] = "completed"
        yield json.dumps(done_event, ensure_ascii=False) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson"
    )


@app.route("/api/story/session/<session_id>", methods=["GET"])
def get_story_session(session_id):
    session = story_sessions.get(session_id)
    if not session:
        return request_error(
            "session_not_found",
            "故事会话不存在或已经失效，请返回首页重新开始。",
            status_code=404,
        )

    state: SimulationState = session["state"]

    return jsonify({
        "session_id": session_id,
        "page": session["page"],
        "isEnd": session["ended"],
        "prompt": session["prompt"],
        "scene": session["scene"],
        "worldview": session["scenario"].public_worldview_markdown,
        "characters": session["public_characters"],
        "turn_count": state.turn_count,
        "current_phase": state.current_phase,
        "world_state": state.public_world_state(),
        "end_reason": state.end_reason,
        "end_kind": state.end_kind,
        "run_status": getattr(state, "run_status", "running"),
        "winner": getattr(state, "winner", ""),
        "public_state": state.public_state_summary(),
        "agents": [
            {
                "name": agent.name,
                "public_profile": agent.public_profile,
                "active": agent.active,
                "alive": agent.alive,
            }
            for agent in state.agents.values()
        ],
        "history": [
            {
                "speaker": msg.speaker,
                "action": msg.action,
                "speech": msg.speech,
                "turn": msg.turn,
                "kind": msg.kind,
                "visibility": msg.visibility,
                "visibility_scopes": msg.scopes,
                "location": msg.location,
                "state_updates": msg.state_updates,
            }
            for msg in state.history
        ],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
