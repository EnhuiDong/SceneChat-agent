import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./HomePage.css";
import { getApiErrorMessage } from "./apiErrors";

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
      const response = await fetch("/api/story/start", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          scene: scene.trim(),
        }),
      });

      const rawText = await response.text();
      let data = null;

      try {
        data = rawText ? JSON.parse(rawText) : null;
      } catch {
        data = null;
      }

      if (!response.ok) {
        openErrorModal(getApiErrorMessage(data, "启动实验失败，请稍后重试。"));
        return;
      }

      const finalScene = data.scene || scene.trim();

      localStorage.setItem("story_prompt", prompt.trim());
      localStorage.setItem("story_scene", finalScene);
      localStorage.setItem("story_batch_size", "10");
      localStorage.setItem("story_session_id", data.session_id);
      localStorage.setItem("story_scenario", JSON.stringify(data.scenario || {}));
      localStorage.setItem("story_pages", JSON.stringify([]));
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
          一句话概念或完整人物、规则与剧情设定，都可以生成可持续推演的场景。
        </p>

        <div className="home-form">
          <label className="home-label">实验设定</label>
          <textarea
            className="home-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="例如：七人狼人杀；也可以直接粘贴完整的人物表、规则、关系和剧情大纲"
            rows={6}
          />

          <label className="home-label">初始场景（可选）</label>
          <input
            className="home-input"
            type="text"
            value={scene}
            onChange={(e) => setScene(e.target.value)}
            placeholder="留空时，系统会根据你的题材自动生成合适的开场"
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
