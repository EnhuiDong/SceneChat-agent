import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./HomePage.css";

function HomePage() {
  const [prompt, setPrompt] = useState("");
  const [scene, setScene] = useState("");
  const [isStarting, setIsStarting] = useState(false);
  const [modalMessage, setModalMessage] = useState("");
  const navigate = useNavigate();

  const openErrorModal = (message) => {
    setModalMessage(message);
  };

  const closeErrorModal = () => {
    setModalMessage("");
  };

  const handleSubmit = async () => {
    if (!prompt.trim()) {
      openErrorModal("请输入实验设定");
      return;
    }

    if (isStarting) return;

    setIsStarting(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/api/story/start", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          scene: scene.trim() || "清晨，星港医疗中心走廊",
          batch_size: 10,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        openErrorModal(data.error || "启动故事失败");
        return;
      }

      const finalScene = scene.trim() || "清晨，星港医疗中心走廊";

      localStorage.setItem("story_prompt", prompt.trim());
      localStorage.setItem("story_scene", finalScene);
      localStorage.setItem("story_batch_size", "10");
      localStorage.setItem("story_session_id", data.session_id);
      localStorage.setItem(
        "story_pages",
        JSON.stringify([
          {
            page: data.page,
            isEnd: data.isEnd,
            hasPlayed: false,
            messages: data.messages,
          },
        ])
      );
      localStorage.setItem("current_page_index", "0");

      navigate("/story", {
        state: {
          prompt: prompt.trim(),
          scene: finalScene,
          sessionId: data.session_id,
        },
      });
    } catch (error) {
      console.error(error);
      openErrorModal("无法连接至后端");
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="home-page">
      <div className="home-overlay" />

      <div className="home-card">
        <div className="home-badge">Social Simulation</div>

        <h1 className="home-title">社会模拟实验设定生成器</h1>
        <p className="home-subtitle">
          输入实验主题与故事起始场景，让角色像小说一样一步步展开对话。
        </p>

        <div className="home-form">
          <label className="home-label">实验设定</label>
          <textarea
            className="home-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="例如：一个高压公司内部关于忠诚、晋升与背叛的社会实验"
            rows={6}
          />

          <label className="home-label">初始场景</label>
          <input
            className="home-input"
            type="text"
            value={scene}
            onChange={(e) => setScene(e.target.value)}
            placeholder="例如：深夜，医院走廊尽头的休息室"
          />

          <button
            className="home-button"
            onClick={handleSubmit}
            disabled={isStarting}
          >
            {isStarting ? (
              <span className="home-button-loading">
                <span className="mini-spinner" />
                正在构建世界...
              </span>
            ) : (
              "开始生成"
            )}
          </button>
        </div>
      </div>

      {modalMessage && (
        <div className="home-modal-overlay" onClick={closeErrorModal}>
          <div className="home-modal" onClick={(e) => e.stopPropagation()}>
            <div className="home-modal-title">提示</div>
            <div className="home-modal-text">{modalMessage}</div>
            <div className="home-modal-actions">
              <button className="home-modal-btn" onClick={closeErrorModal}>
                我知道了
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default HomePage;