import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { getApiErrorMessage } from "./apiErrors";
import { saveStorySetup } from "./scenarioStorage";
import { startStoryBuild, cancelStoryBuild, fetchStoryBuildStatus } from "./storyApi";
import "./BuildReview.css";

const BUILD_STAGES = [
  ["embedding_preflight", "检查向量模型", "开始生成前确认知识检索服务可用"],
  ["generation_preflight", "检查模型", "确认生成服务和 JSON 能力"],
  ["brief", "理解设定", "提取硬约束、题材和信息边界"],
  ["world", "构建世界", "生成场景、规则、阶段和结束条件"],
  ["characters", "生成角色", "创建公开档案和隔离的私密上下文"],
  ["validation", "一致性校验", "核对人数、约束、可见性和规则"],
  ["runtime", "准备模拟", "建立角色状态、知识边界和推演客户端"],
  ["storage", "保存档案", "归档完整结构化实验"],
];

const PENDING_BUILD_KEY = "story_pending_build";

function pendingBuild(prompt, scene) {
  try {
    const saved = JSON.parse(localStorage.getItem(PENDING_BUILD_KEY) || "null");
    return saved?.prompt === prompt && saved?.scene === scene ? saved : null;
  } catch { return null; }
}

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
  const [activity, setActivity] = useState(null);
  const [cancelling, setCancelling] = useState(false);
  const buildIdRef = useRef(pendingBuild(prompt, scene)?.buildId || null);
  const [canResume, setCanResume] = useState(Boolean(buildIdRef.current));
  const controllerRef = useRef(null);
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
    const pending = pendingBuild(prompt, scene);
    if (runId === 1 && pending?.resumable === false) {
      setCanResume(false);
      setErrorMessage(pending.message || "上次修复未取得进展，请修改设定或模型后重新生成。");
      let active = true;
      if (pending.buildId) fetchStoryBuildStatus(pending.buildId).then((status) => {
        if (active && status.recovery_available) {
          setCanResume(true);
          setErrorMessage("修复器已更新，可以从保留的世界与角色检查点继续；点击后才会调用模型。");
        }
      }).catch(() => {});
      return () => { active = false; };
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    cancellingRef.current = false;
    let settled = false;
    let dispatched = false;
    let activeBuildId = buildIdRef.current;

    // Deferring dispatch lets StrictMode clean up its probe effect without
    // creating a second paid build request.
    const dispatch = setTimeout(() => {
    dispatched = true;
    startStoryBuild({
      prompt,
      scene,
      buildId: buildIdRef.current,
      signal: controller.signal,
      onEvent: async (event) => {
        if (cancellingRef.current || controller.signal.aborted) return;
        if (event.type === "build_started") {
          activeBuildId = event.build_id;
          buildIdRef.current = event.build_id;
          setCanResume(true);
          localStorage.setItem(PENDING_BUILD_KEY, JSON.stringify({ buildId: event.build_id, prompt, scene }));
        }
        if (event.type === "build_activity") setActivity(event);
        if (event.type === "error") {
          setCanResume(event.resumable !== false);
          if (event.resumable === false) {
            // Retain the server checkpoint for inspection, but do not silently
            // resume the same failed repair when this page mounts again.
            localStorage.setItem(PENDING_BUILD_KEY, JSON.stringify({
              buildId: event.build_id, prompt, scene, resumable: false,
              message: event.error?.message,
            }));
          }
        }
        if (event.type === "build_progress") {
          setStageState((previous) => ({
            ...previous,
            [event.stage]: event,
          }));
        }
        if (event.type === "story_ready") {
          settled = true;
          localStorage.removeItem(PENDING_BUILD_KEY);
          buildIdRef.current = null;
          setCanResume(false);
          saveStorySetup(localStorage, prompt, event.data);
          await new Promise((resolve) => setTimeout(resolve, 350));
          if (!controller.signal.aborted) navigate("/review", { replace: true });
        }
      },
    }).catch((error) => {
      if (controller.signal.aborted) return;
      if (error.name === "AbortError") {
        if (!cancellingRef.current) setErrorMessage("生成已暂停，可以返回修改后重新开始。");
        return;
      }
      setErrorMessage(getApiErrorMessage(error, error.message || "场景生成失败。"));
    }).finally(() => { settled = true; });
    }, 0);
    return () => {
      clearTimeout(dispatch);
      controller.abort();
      if (dispatched && !settled && activeBuildId) cancelStoryBuild(activeBuildId).catch(() => {});
    };
  }, [navigate, prompt, runId, scene]);

  const cancelBuild = async () => {
    setCancelling(true);
    cancellingRef.current = true;
    try {
      if (buildIdRef.current) await cancelStoryBuild(buildIdRef.current);
      controllerRef.current?.abort();
      navigate("/", { replace: true, state: { prompt, scene } });
    } catch (error) {
      cancellingRef.current = false;
      setErrorMessage(error.message);
    } finally { setCancelling(false); }
  };

  const retryBuild = async (restart = false) => {
    if (restart && buildIdRef.current) {
      try { await cancelStoryBuild(buildIdRef.current); }
      catch (error) { setErrorMessage(error.message); return; }
      buildIdRef.current = null;
      localStorage.removeItem(PENDING_BUILD_KEY);
      setCanResume(false);
    }
    controllerRef.current?.abort();
    setStageState({});
    setActivity(null);
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

        {activity && <div className="build-activity" aria-live="polite">
          <strong>{BUILD_STAGES.find(([id]) => id === activity.stage)?.[1] || "构建中"}</strong>
          <span>已用 {Math.floor(activity.elapsed_seconds)} 秒 · 当前预算剩余 {Math.ceil(activity.remaining_seconds)} 秒</span>
          {activity.request_attempt && <span>本步骤请求 {activity.request_attempt}/{activity.request_limit} · JSON 修复 {activity.json_repair_attempt} 次 · 传输重试 {activity.transport_retry} 次</span>}
          {activity.reason && <small>{activity.reason}</small>}
          {activity.issues?.length > 0 && <details>
            <summary>查看本次修复问题（可能涉及隐藏设定） · 第 {activity.repair_attempt} 次修复</summary>
            <ul>{activity.issues.map((issue, index) => <li key={index}>{issue}</li>)}</ul>
          </details>}
          <small>已完成的世界、角色会保存在本地检查点；超时后无需全部重建。</small>
        </div>}

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
                    ? errorMessage ? "已停止" : "进行中"
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
            {canResume && <button type="button" onClick={() => retryBuild()}>从检查点继续</button>}
            <button type="button" onClick={() => retryBuild(true)}>重新生成全部</button>
          </div>
        ) : null}

        <button className="flow-text-button" type="button" onClick={cancelBuild} disabled={cancelling}>
          {cancelling ? "正在通知后端停止…" : "取消并返回修改（保留检查点）"}
        </button>
      </section>
    </main>
  );
}

export default BuildPage;
