import os
import uuid
import json
from typing import Dict

from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS

from World import generate_worldview
from Character import generate_characters
from history import get_index, SimulationState, simulate_next_turn, Message

app = Flask(__name__)
CORS(app)

story_sessions: Dict[str, dict] = {}

DEFAULT_BATCH_SIZE = 10
MAX_TURNS = 30



def message_to_frontend(msg: Message):
    return {
        "id": msg.turn,
        "speaker": msg.speaker,
        "action": msg.action,
        "speech": msg.speech,
        "display_text": msg.speech,
        "turn": msg.turn,
    }


@app.route("/api/story/start", methods=["POST"])
def start_story():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    user_prompt = (data.get("prompt") or "").strip()
    initial_scene = (data.get("scene") or "").strip()

    if not user_prompt:
        return jsonify({"error": "prompt is required"}), 400

    if not initial_scene:
        initial_scene = "清晨，星港医疗中心走廊"

    try:
        worldview = generate_worldview(user_prompt)
        characters = generate_characters(user_prompt, worldview)

        os.makedirs("./data", exist_ok=True)

        with open("./data/world.md", "w", encoding="utf-8") as f:
            f.write(worldview)

        with open("./data/character.md", "w", encoding="utf-8") as f:
            f.write(characters)

        index = get_index()
        rag_retriever = index.as_retriever(similarity_top_k=3)

        state = SimulationState(initial_scene)

        session_id = str(uuid.uuid4())
        story_sessions[session_id] = {
            "prompt": user_prompt,
            "scene": initial_scene,
            "worldview": worldview,
            "characters": characters,
            "state": state,
            "rag_retriever": rag_retriever,
            "page": 0,         # 注意：现在从 0 开始，还没正式生成第一页
            "ended": False,
        }

        return jsonify({
            "session_id": session_id,
            "page": 0,
            "isEnd": False,
            "worldview": worldview,
            "characters": characters,
            "scene": initial_scene,
            "messages": [],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/story/next-stream", methods=["POST"])
def next_story_page_stream():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    session = story_sessions.get(session_id)
    if not session:
        return jsonify({"error": "session not found"}), 404

    if session["ended"]:
        return jsonify({
            "session_id": session_id,
            "page": session["page"],
            "isEnd": True,
            "messages": [],
        }), 200

    batch_size = data.get("batch_size", DEFAULT_BATCH_SIZE)
    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = DEFAULT_BATCH_SIZE

    state: SimulationState = session["state"]
    rag_retriever = session["rag_retriever"]
    next_page_number = session["page"] + 1

    def generate():
        sent_count = 0

        yield json.dumps({
            "type": "page_start",
            "page": next_page_number,
            "session_id": session_id,
        }, ensure_ascii=False) + "\n"

        for _ in range(batch_size):
            msg = simulate_next_turn(state, rag_retriever)
            if msg is None:
                continue

            state.add_message(msg)
            sent_count += 1

            yield json.dumps({
                "type": "message",
                "page": next_page_number,
                "message": message_to_frontend(msg),
            }, ensure_ascii=False) + "\n"

        session["page"] = next_page_number

        is_end = state.turn_count >= MAX_TURNS
        session["ended"] = is_end

        yield json.dumps({
            "type": "page_done",
            "page": next_page_number,
            "session_id": session_id,
            "isEnd": is_end,
            "count": sent_count,
        }, ensure_ascii=False) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson"
    )


@app.route("/api/story/session/<session_id>", methods=["GET"])
def get_story_session(session_id):
    session = story_sessions.get(session_id)
    if not session:
        return jsonify({"error": "session not found"}), 404

    state: SimulationState = session["state"]

    return jsonify({
        "session_id": session_id,
        "page": session["page"],
        "isEnd": session["ended"],
        "prompt": session["prompt"],
        "scene": session["scene"],
        "worldview": session["worldview"],
        "characters": session["characters"],
        "turn_count": state.turn_count,
        "history": [
            {
                "speaker": msg.speaker,
                "action": msg.action,
                "speech": msg.speech,
                "turn": msg.turn,
            }
            for msg in state.history
        ],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)