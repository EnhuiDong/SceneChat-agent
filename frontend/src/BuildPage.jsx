import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { getApiErrorMessage } from "./apiErrors";
import { saveStorySetup } from "./scenarioStorage";
import { startStoryBuild } from "./storyApi";
import "./BuildReview.css";

const BUILD_STAGES = [
  ["generation_preflight", "检查模型", "确认生成服务和 JSON 能力"],
  ["brief", "理解设定", "提取硬约束、题材和信息边界"],
  ["world", "构建世界", "生成场景、规则、阶段和结束条件"],
  ["characters", "生成角色", "创建公开档案和隔离的私密上下文"],
  ["validation", "一致性校验", "核对人数、约束、可见性和规则"],
  ["embedding_preflight", "准备知识", "仅在长背景需要 RAG 时检查向量模型"],
  ["runtime", "准备模拟", "建立角色状态、知识边界和推演客户端"],
  ["storage", "保存档案", "归档完整结构化实验"],
];

function BuildPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const prompt =
    location.state?.prompt || localStorage.getItem("story_draft_prompt") || "";
  const scene =
    location.state?.scene || localStorage.getItem("story_draft_scene") || "";
  const [runId, setRunId] = useState(1);
  const [stageState, setStageState] = useState({});
  const [errorMessage, setErrorMessage] = useState("");
  const controllerRef = useRef(null);
  const startedRunRef = useRef(0);
  const cancellingRef = useRef(false);

  const completedCount = useMemo(
    () =>
      BUILD_STAGES.filter(([id]) =>
        ["completed", "skipped"].includes(stageState[id]?.status)
      ).length,
    [stageState]
  );

  useEffect(() => {
    if (!prompt) {
      navigate("/", { replace: true });
      return;
    }
    if (startedRunRef.current === runId) return;
    startedRunRef.current = runId;
    const controller = new AbortController();
    controllerRef.current = controller;
    cancellingRef.current = false;

    startStoryBuild({
      prompt,
      scene,
      signal: controller.signal,
      onEvent: async (event) => {
        if (event.type === "build_progress") {
          setStageState((previous) => ({
            ...previous,
            [event.stage]: event,
          }));
        }
        if (event.type === "story_ready") {
          saveStorySetup(localStorage, prompt, event.data);
          await new Promise((resolve) => setTimeout(resolve, 350));
          navigate("/review", { replace: true });
        }
      },
    }).catch((error) => {
      if (error.name === "AbortError") {
        if (!cancellingRef.current) setErrorMessage("生成已暂停，可以返回修改后重新开始。");
        return;
      }
      setErrorMessage(getApiErrorMessage(error, error.message || "场景生成失败。"));
    });
  }, [navigate, prompt, runId, scene]);

  const cancelBuild = () => {
    cancellingRef.current = true;
    controllerRef.current?.abort();
    navigate("/", {
      replace: true,
      state: { prompt, scene },
    });
  };

  const retryBuild = () => {
    controllerRef.current?.abort();
    setStageState({});
    setErrorMessage("");
    setRunId((value) => value + 1);
  };

  return (
    <main className="flow-page build-page">
      <section className="flow-card build-card" aria-live="polite">
        <div className="flow-eyebrow">SCENE CONSTRUCTION</div>
        <div className="build-heading-row">
          <div>
            <h1>正在把设定变成可运行的世界</h1>
            <p>每一项都来自真实的后端阶段，不会用假进度掩盖等待。</p>
          </div>
          <div className="progress-orbit" aria-label={`完成 ${completedCount} 个阶段`}>
            <strong>{completedCount}</strong>
            <span>/ {BUILD_STAGES.length}</span>
          </div>
        </div>

        <div className="build-prompt-preview">{prompt}</div>

        <ol className="build-stage-list">
          {BUILD_STAGES.map(([id, title, description], index) => {
            const event = stageState[id];
            const status = event?.status || "pending";
            return (
              <li className={`build-stage ${status}`} key={id}>
                <span className="stage-index">
                  {status === "completed" ? "✓" : status === "skipped" ? "—" : index + 1}
                </span>
                <span className="stage-copy">
                  <strong>{title}</strong>
                  <small>{event?.reason || description}</small>
                </span>
                <span className="stage-status">
                  {status === "started"
                    ? "进行中"
                    : status === "completed"
                    ? "完成"
                    : status === "skipped"
                    ? "无需执行"
                    : "等待"}
                </span>
              </li>
            );
          })}
        </ol>

        {errorMessage ? (
          <div className="flow-error" role="alert">
            <strong>生成未完成</strong>
            <span>{errorMessage}</span>
            <button type="button" onClick={retryBuild}>从头重试</button>
          </div>
        ) : null}

        <button className="flow-text-button" type="button" onClick={cancelBuild}>
          取消并返回修改
        </button>
      </section>
    </main>
  );
}

export default BuildPage;
