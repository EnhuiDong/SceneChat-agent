import os
import uuid
from dataclasses import asdict
from typing import Dict, List

from flask import Flask, jsonify, request
from flask_cors import CORS

from World import generate_worldview
from Character import generate_characters
from history import get_index, SimulationState, simulate_next_turn, Message

app = Flask(__name__)
CORS(app)

# 简单内存 session 存储
# 以后可以换成 Redis / 数据库
story_sessions: Dict[str, dict] = {}

DEFAULT_BATCH_SIZE = 10
MAX_TURNS = 30  # 你可以自己改成更长，或者改成让模型判断结束


def message_to_frontend(msg: Message, current_user_speaker: str = None):
    """
    把你的 Message 转成前端想要的格式
    """
    print( "id:", msg.turn,"speaker:",msg.speaker,"action:",msg.action,"text:", msg.speech,"turn:", msg.turn)
    return {
        "id": msg.turn,
        "speaker": msg.speaker,
        "action": msg.action,
        "text": msg.speech,
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
        # 1. 生成世界观
        worldview = generate_worldview(user_prompt)

        # 2. 生成角色设定
        characters = generate_characters(user_prompt, worldview)

        os.makedirs("./data", exist_ok=True)

        with open("./data/world.md", "w", encoding="utf-8") as f:
            f.write(worldview)

        with open("./data/character.md", "w", encoding="utf-8") as f:
            f.write(characters)

        # 3. 加载索引
        index = get_index()
        rag_retriever = index.as_retriever(similarity_top_k=3)

        # 4. 初始化故事状态
        state = SimulationState(initial_scene)

        # 5. 先生成第一页（默认 5 轮）
        batch_size = data.get("batch_size", DEFAULT_BATCH_SIZE)
        try:
            batch_size = int(batch_size)
        except Exception:
            batch_size = DEFAULT_BATCH_SIZE

        page_messages: List[Message] = []

        for _ in range(batch_size):
            msg = simulate_next_turn(state, rag_retriever)
            if msg is None:
                continue
            state.add_message(msg)
            page_messages.append(msg)

        # 6. 创建 session
        session_id = str(uuid.uuid4())
        story_sessions[session_id] = {
            "prompt": user_prompt,
            "scene": initial_scene,
            "worldview": worldview,
            "characters": characters,
            "state": state,
            "rag_retriever": rag_retriever,
            "page": 1,
            "ended": False,
        }

        # 7. 如果一开始就没生成出内容，也要返回
        frontend_messages = [message_to_frontend(m) for m in page_messages]

        return jsonify({
            "session_id": session_id,
            "page": 1,
            "isEnd": False,
            "worldview": worldview,
            "characters": characters,
            "scene": initial_scene,
            "messages": frontend_messages,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/story/next", methods=["POST"])
def next_story_page():
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
        })

    try:
        batch_size = data.get("batch_size", DEFAULT_BATCH_SIZE)
        try:
            batch_size = int(batch_size)
        except Exception:
            batch_size = DEFAULT_BATCH_SIZE

        state: SimulationState = session["state"]
        rag_retriever = session["rag_retriever"]

        page_messages: List[Message] = []

        for _ in range(batch_size):
            msg = simulate_next_turn(state, rag_retriever)
            if msg is None:
                continue
            state.add_message(msg)
            page_messages.append(msg)

        session["page"] += 1

        # 暂时先用“轮数达到上限”判断故事结束
        # 以后你可以改成模型返回 END
        is_end = state.turn_count >= MAX_TURNS
        session["ended"] = is_end

        frontend_messages = [message_to_frontend(m) for m in page_messages]

        return jsonify({
            "session_id": session_id,
            "page": session["page"],
            "isEnd": is_end,
            "messages": frontend_messages,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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