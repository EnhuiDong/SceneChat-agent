import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { clearStoryStorage, saveStorySetup } from "./scenarioStorage";
import { clearStorySessions, deleteStorySession, fetchStorySession, listStorySessions } from "./storyApi";
import "./HomePage.css";

const EXAMPLES = [
  "七人狼人杀，角色自行发言、投票并推进昼夜阶段",
  "三位合租室友因为房租分摊爆发争执，每个人都有难言之隐",
  "封闭列车上的推理局：六名乘客中有人隐瞒了昨夜的行踪",
];

function HomePage() {
  const location = useLocation();
  const initialPrompt =
    location.state?.prompt || localStorage.getItem("story_draft_prompt") || "";
  const initialScene =
    location.state?.scene || localStorage.getItem("story_draft_scene") || "";
  const [prompt, setPrompt] = useState(initialPrompt);
  const [scene, setScene] = useState(initialScene);
  const [modalMessage, setModalMessage] = useState("");
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyBusy, setHistoryBusy] = useState("");
  const [confirmAction, setConfirmAction] = useState(null);
  const navigate = useNavigate();
  const inputMode = useMemo(() => {
    const trimmed = prompt.trim();
    return trimmed.length >= 180 || trimmed.split(/\r?\n/).filter(Boolean).length >= 5
      ? "完整设定"
      : "概念设定";
  }, [prompt]);

  useEffect(() => {
    let active = true;
    listStorySessions()
      .then((items) => { if (active) setHistory(items); })
      .catch((error) => { if (active) setModalMessage(error.message); })
      .finally(() => { if (active) setHistoryLoading(false); });
    return () => { active = false; };
  }, []);

  const openErrorModal = (message) => {
    setModalMessage(message);
  };

  const closeErrorModal = () => {
    setModalMessage("");
  };

  const handleSubmit = () => {
    if (!prompt.trim()) {
      openErrorModal("请输入实验设定");
      return;
    }
    const cleanPrompt = prompt.trim();
    const cleanScene = scene.trim();
    localStorage.setItem("story_draft_prompt", cleanPrompt);
    localStorage.setItem("story_draft_scene", cleanScene);
    navigate("/build", { state: { prompt: cleanPrompt, scene: cleanScene } });
  };

  const handleKeyDown = (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      handleSubmit();
    }
  };

  const continueStory = async (item) => {
    if (historyBusy) return;
    setHistoryBusy(item.id);
    try {
      const snapshot = await fetchStorySession(item.id);
      saveStorySetup(localStorage, snapshot.prompt, snapshot);
      navigate(snapshot.page > 0 ? "/story" : "/review");
    } catch (error) {
      openErrorModal(error.message || "恢复推演失败。");
    } finally {
      setHistoryBusy("");
    }
  };

  const runConfirmedAction = async () => {
    if (!confirmAction || historyBusy) return;
    const action = confirmAction;
    setConfirmAction(null);
    setHistoryBusy(action.type === "clear" ? "all" : action.session.id);
    try {
      if (action.type === "clear") {
        await clearStorySessions();
        clearStoryStorage(localStorage);
        setHistory([]);
      } else {
        await deleteStorySession(action.session.id);
        if (localStorage.getItem("story_session_id") === action.session.id) {
          clearStoryStorage(localStorage);
        }
        setHistory((items) => items.filter((item) => item.id !== action.session.id));
      }
    } catch (error) {
      openErrorModal(error.message || "操作失败，请稍后重试。");
    } finally {
      setHistoryBusy("");
    }
  };

  return (
    <div className="home-page">
      <div className="home-overlay" />

      <section className="home-card">
        <div className="home-badge">SCENECHAT · SOCIAL SIMULATION</div>

        <h1 className="home-title">把一个想法，变成会自己发展的世界</h1>
        <p className="home-subtitle">
          从一句话到完整人物表、规则和剧情大纲，都能转化为可持续推演的多角色场景。
        </p>

        <div className="home-flow" aria-label="创建流程">
          <span className="active">01 写下设定</span><i />
          <span>02 审阅场景</span><i />
          <span>03 开始推演</span>
        </div>

        <form className="home-form" onSubmit={(event) => { event.preventDefault(); handleSubmit(); }}>
          <div className="home-label-row">
            <label className="home-label" htmlFor="scenario-prompt">你的设定</label>
            <span className={`input-mode ${inputMode === "完整设定" ? "detailed" : ""}`}>
              {inputMode} · {prompt.length} 字
            </span>
          </div>
          <textarea
            id="scenario-prompt"
            className="home-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="一句话也可以，例如：七十年代王案杀。也可以直接粘贴完整的人物、关系、规则、场地和剧情细节。"
            rows={9}
          />

          <div className="example-row" aria-label="示例设定">
            <span>试试：</span>
            {EXAMPLES.map((example, index) => (
              <button type="button" key={example} onClick={() => setPrompt(example)}>
                示例 {index + 1}
              </button>
            ))}
          </div>

          <details className="home-advanced" open={Boolean(scene)}>
            <summary>补充开场位置或时刻 <span>可选</span></summary>
            <label className="sr-only" htmlFor="opening-scene">初始场景</label>
            <input
              id="opening-scene"
              className="home-input"
              type="text"
              value={scene}
              onChange={(e) => setScene(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="例如：暴雨停电后的客厅。留空会由系统生成。"
            />
          </details>

          <button
            className="home-button"
            type="submit"
          >
            构建场景 <span aria-hidden="true">→</span>
          </button>
          <small className="home-shortcut">Ctrl / ⌘ + Enter 快速开始</small>
        </form>

        <section className="story-library" aria-labelledby="story-library-title">
          <header><div><span>LOCAL ARCHIVE</span><h2 id="story-library-title">历史推演</h2></div>{history.length ? <button type="button" className="clear-history" onClick={() => setConfirmAction({ type: "clear" })}>清除全部</button> : null}</header>
          {historyLoading ? <p className="library-empty">正在读取本机记录…</p> : history.length ? (
            <div className="story-history-list">{history.map((item) => (
              <article key={item.id}>
                <div className="history-copy"><strong>{item.title}</strong><p>{item.prompt}</p><span>{item.turn_count} 回合 · {item.status === "ended" ? "已结束" : item.status === "blocked" ? "已中断" : "可继续"} · {new Date(item.updated_at).toLocaleString()}</span></div>
                <div className="history-actions"><button type="button" onClick={() => continueStory(item)} disabled={Boolean(historyBusy)}>{historyBusy === item.id ? "载入中…" : "继续"}</button><button type="button" className="delete-history" onClick={() => setConfirmAction({ type: "delete", session: item })} disabled={Boolean(historyBusy)} aria-label={`删除 ${item.title}`}>删除</button></div>
              </article>
            ))}</div>
          ) : <p className="library-empty">还没有保存的推演。创建后会自动保存在这台设备上。</p>}
        </section>
      </section>

      {modalMessage && (
        <div className="home-modal-overlay" onClick={closeErrorModal}>
          <div className="home-modal" role="alertdialog" aria-modal="true" aria-labelledby="home-modal-title" onClick={(e) => e.stopPropagation()}>
            <div className="home-modal-title" id="home-modal-title">提示</div>
            <div className="home-modal-text">{modalMessage}</div>
            <div className="home-modal-actions">
              <button className="home-modal-btn" onClick={closeErrorModal}>
                我知道了
              </button>
            </div>
          </div>
        </div>
      )}

      {confirmAction && (
        <div className="home-modal-overlay" onClick={() => setConfirmAction(null)}>
          <div className="home-modal" role="alertdialog" aria-modal="true" aria-labelledby="confirm-modal-title" onClick={(event) => event.stopPropagation()}>
            <div className="home-modal-title" id="confirm-modal-title">{confirmAction.type === "clear" ? "清除全部推演？" : "删除这次推演？"}</div>
            <div className="home-modal-text">{confirmAction.type === "clear" ? "所有历史推演、人物状态和事件记录都会从本机删除，且无法撤销。" : `“${confirmAction.session.title}”及其全部人物和历史记录会被永久删除。`}</div>
            <div className="home-modal-actions confirm-actions"><button type="button" className="cancel-confirm" onClick={() => setConfirmAction(null)}>取消</button><button type="button" className="danger-confirm" onClick={runConfirmedAction}>确认删除</button></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default HomePage;
