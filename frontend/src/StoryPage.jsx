import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getApiErrorMessage, readApiError } from "./apiErrors";
import { clearStoryStorage, loadStoredScenario } from "./scenarioStorage";
import {
  cancelStoryIntervention,
  confirmStoryIntervention,
  fetchStoryExport,
  fetchStorySession,
  previewStoryIntervention,
  updateStoryPace,
} from "./storyApi";
import { loadPageIndex, loadStoryPages } from "./storyStorage";
import "./StoryPage.css";

const ROLE_COLORS = ["#44705a", "#8a5d3b", "#596b98", "#8b536b", "#6f6740", "#4f7180", "#795487", "#89704c"];

function roleColor(name = "") {
  const hash = [...name].reduce((sum, character) => sum + character.charCodeAt(0), 0);
  return ROLE_COLORS[hash % ROLE_COLORS.length];
}

function MessageCard({ message, skipToken }) {
  const fullText = message.display_text || "";
  const instant = Boolean(message.replayed || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches || !fullText);
  const [visibleText, setVisibleText] = useState(instant ? fullText : "");
  const [initialSkipToken] = useState(skipToken);

  useEffect(() => {
    if (instant) return undefined;
    let index = 0;
    const timer = window.setInterval(() => {
      index = Math.min(index + 2, fullText.length);
      setVisibleText(fullText.slice(0, index));
      if (index >= fullText.length) window.clearInterval(timer);
    }, 18);
    return () => window.clearInterval(timer);
  }, [fullText, message.id, instant]);

  const renderedText = skipToken !== initialSkipToken ? fullText : visibleText;

  const narration = ["narration", "intervention"].includes(message.kind);
  const kindLabel = message.kind === "intervention" ? "导演干预" : message.kind === "narration" ? "旁白" : "角色行动";
  return (
    <article className={`timeline-message ${narration ? "narration" : "dialogue"} ${message.kind === "intervention" ? "director-event" : ""}`} style={{ "--role-color": roleColor(message.speaker) }}>
      <header><span className="speaker-dot" /> <strong>{message.speaker || "旁白"}</strong><small>{kindLabel}</small></header>
      {message.action ? <p className="message-action">{message.action}</p> : null}
      <p>{renderedText}{renderedText.length < fullText.length ? <span className="typing-caret" /> : null}</p>
    </article>
  );
}

function StoryPage() {
  const navigate = useNavigate();
  const sessionId = localStorage.getItem("story_session_id") || "";
  const prompt = localStorage.getItem("story_prompt") || "";
  const scene = localStorage.getItem("story_scene") || "";
  const storedScenario = useMemo(() => loadStoredScenario(localStorage), []);
  const initialPages = useMemo(() => loadStoryPages(localStorage), []);
  const [pages, setPages] = useState(initialPages);
  const [currentPageIndex, setCurrentPageIndex] = useState(() => loadPageIndex(localStorage, initialPages.length));
  const [snapshot, setSnapshot] = useState({ scenario: storedScenario, agents: storedScenario.characters || [], world_state: storedScenario.initial_state || {}, current_phase: storedScenario.phases?.[0] || "准备", turn_count: 0, revision: 0, run_status: "running" });
  const [snapshotReady, setSnapshotReady] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [pausedPage, setPausedPage] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [showQuitModal, setShowQuitModal] = useState(false);
  const [selectedCharacterId, setSelectedCharacterId] = useState("");
  const [skipToken, setSkipToken] = useState(0);
  const [autoAdvance, setAutoAdvance] = useState(false);
  const [autoCountdown, setAutoCountdown] = useState(null);
  const [batchSize, setBatchSize] = useState(() => Number(localStorage.getItem("story_batch_size")) || 6);
  const [interventionDraft, setInterventionDraft] = useState("");
  const [interventionScope, setInterventionScope] = useState("next_scene");
  const [interventionDuration, setInterventionDuration] = useState(6);
  const [interventionPreview, setInterventionPreview] = useState(null);
  const [isDirecting, setIsDirecting] = useState(false);
  const [paceDraft, setPaceDraft] = useState(50);
  const streamAbortRef = useRef(null);
  const streamReaderRef = useRef(null);
  const intentionalAbortRef = useRef(false);
  const initialRunRef = useRef(false);
  const contentRef = useRef(null);

  const currentPage = pages[currentPageIndex] || { page: 1, messages: [], isEnd: false };
  const scenario = snapshot.scenario || storedScenario;
  const agents = useMemo(() => {
    const publicCharacters = scenario.characters || [];
    const runtimeAgents = snapshot.agents?.length ? snapshot.agents : publicCharacters;
    return runtimeAgents.map((agent) => ({
      ...publicCharacters.find((character) => character.id === agent.id || character.name === agent.name),
      ...agent,
    }));
  }, [scenario.characters, snapshot.agents]);
  const selectedCharacter = agents.find((agent) => (agent.id || agent.name) === selectedCharacterId);
  const pendingInterventions = (snapshot.interventions || []).filter((item) => {
    if (item.status === "pending") return true;
    if (item.status !== "applied" || item.mode !== "guidance") return false;
    if (item.scope === "persistent") return true;
    return item.scope === "turns" && snapshot.turn_count - item.applied_at_turn < (item.expires_after_turns || 1);
  });
  const progressPercent = Math.round((snapshot.arc_state?.progress || 0) * 100);

  useEffect(() => { localStorage.setItem("story_pages", JSON.stringify(pages)); }, [pages]);
  useEffect(() => { localStorage.setItem("current_page_index", String(currentPageIndex)); }, [currentPageIndex]);
  useEffect(() => { localStorage.setItem("story_batch_size", String(batchSize)); }, [batchSize]);
  useEffect(() => { setPaceDraft(snapshot.arc_state?.pace ?? 50); }, [snapshot.arc_state?.pace]);
  useEffect(() => { contentRef.current?.scrollTo({ top: contentRef.current.scrollHeight, behavior: "smooth" }); }, [currentPage.messages?.length, isGenerating]);

  const syncSnapshot = useCallback(async () => {
    if (!sessionId) return;
    try {
      setSnapshot(await fetchStorySession(sessionId));
    } catch (error) {
      console.warn("Unable to refresh public session snapshot", error);
    } finally {
      setSnapshotReady(true);
    }
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId || !prompt) {
      navigate("/", { replace: true });
      return;
    }
    syncSnapshot();
  }, [navigate, prompt, sessionId, syncSnapshot]);

  const generatePage = useCallback(async (targetIndex, pageNumber) => {
    if (isGenerating || !sessionId) return;
    setIsGenerating(true);
    setErrorMessage("");
    setPausedPage(null);
    intentionalAbortRef.current = false;

    const existingPage = pages[targetIndex];
    const requestId = existingPage?.requestId || globalThis.crypto?.randomUUID?.() || `${sessionId}-${pageNumber}-${Date.now()}`;
    const baseMessages = existingPage?.messages || [];
    const pendingPage = { page: pageNumber, messages: baseMessages, isEnd: false, requestId, runStatus: "running" };
    setPages((previous) => { const next = [...previous]; next[targetIndex] = pendingPage; return next; });
    setCurrentPageIndex(targetIndex);
    localStorage.setItem("story_pending_page_request", JSON.stringify({ requestId, page: pageNumber }));

    try {
      const controller = new AbortController();
      streamAbortRef.current = controller;
      const response = await fetch("/api/story/next-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, batch_size: batchSize, request_id: requestId, expected_page: pageNumber, expected_revision: snapshot.revision }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await readApiError(response, "继续推演失败，请稍后重试。"));
      if (!response.body) throw new Error("后端没有返回可读取的推演流。");
      const reader = response.body.getReader();
      streamReaderRef.current = reader;
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      const consume = (line) => {
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.type === "error") throw new Error(getApiErrorMessage(event, "推演中断。"));
        if (event.type === "message") {
          const nextMessage = { ...event.message, replayed: Boolean(event.replayed) };
          setPages((previous) => previous.map((page, index) => index !== targetIndex ? page : page.messages.some((item) => (item.event_id && item.event_id === nextMessage.event_id) || (!item.event_id && item.id === nextMessage.id)) ? page : { ...page, messages: [...page.messages, nextMessage] }));
        }
        if (event.type === "page_done") {
          localStorage.removeItem("story_pending_page_request");
          setPages((previous) => previous.map((page, index) => index === targetIndex ? { ...page, isEnd: event.isEnd, endReason: event.end_reason || "", endKind: event.end_kind || "", runStatus: event.run_status || "running" } : page));
        }
      };
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        lines.forEach(consume);
      }
      if (buffer.trim()) consume(buffer);
      await syncSnapshot();
    } catch (error) {
      if (error.name === "AbortError" && intentionalAbortRef.current) {
        setPausedPage({ index: targetIndex, page: pageNumber });
      } else {
        console.error(error);
        setPausedPage({ index: targetIndex, page: pageNumber });
        setErrorMessage(error.message || "推演中断，请继续或重试。");
      }
    } finally {
      streamReaderRef.current = null;
      streamAbortRef.current = null;
      setIsGenerating(false);
    }
  }, [batchSize, isGenerating, pages, sessionId, snapshot.revision, syncSnapshot]);

  useEffect(() => {
    if (!sessionId || !snapshotReady || pages.length || initialRunRef.current) return;
    initialRunRef.current = true;
    generatePage(0, 1);
  }, [generatePage, pages.length, sessionId, snapshotReady]);

  useEffect(() => {
    const directorBusy = interventionDraft.trim() || interventionPreview || isDirecting;
    if (!autoAdvance || isGenerating || pausedPage || directorBusy || !currentPage.messages?.length || currentPage.isEnd) {
      setAutoCountdown(null);
      return undefined;
    }
    let remaining = 3;
    setAutoCountdown(remaining);
    const timer = window.setInterval(() => {
      remaining -= 1;
      setAutoCountdown(remaining);
      if (remaining <= 0) {
        window.clearInterval(timer);
        generatePage(currentPageIndex + 1, currentPage.page + 1);
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [autoAdvance, currentPage.isEnd, currentPage.messages?.length, currentPage.page, currentPageIndex, generatePage, interventionDraft, interventionPreview, isDirecting, isGenerating, pausedPage]);

  const pauseGeneration = () => {
    intentionalAbortRef.current = true;
    setPausedPage({ index: currentPageIndex, page: currentPage.page });
    setIsGenerating(false);
    streamReaderRef.current?.cancel().catch(() => {});
    streamAbortRef.current?.abort();
  };

  const nextPage = () => {
    if (isGenerating) return;
    if (pages[currentPageIndex + 1]) setCurrentPageIndex((value) => value + 1);
    else if (!currentPage.isEnd) generatePage(currentPageIndex + 1, currentPage.page + 1);
  };

  const confirmQuit = async () => {
    streamAbortRef.current?.abort();
    clearStoryStorage(localStorage);
    navigate("/", { replace: true });
  };

  const exportStory = async () => {
    if (!sessionId || isExporting) return;
    setIsExporting(true);
    setErrorMessage("");
    try {
      const { blob, filename } = await fetchStoryExport(sessionId);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      setErrorMessage(error.message || "导出完整档案失败。");
    } finally {
      setIsExporting(false);
    }
  };

  const previewIntervention = async () => {
    if (!interventionDraft.trim() || isDirecting || isGenerating) return;
    setIsDirecting(true);
    setErrorMessage("");
    try {
      const payload = await previewStoryIntervention(sessionId, {
        text: interventionDraft,
        scope: interventionScope,
        expires_after_turns: interventionDuration,
        expected_revision: snapshot.revision,
      });
      setInterventionPreview(payload.intervention);
    } catch (error) {
      setErrorMessage(error.message || "无法预检剧情干预。");
      await syncSnapshot();
    } finally {
      setIsDirecting(false);
    }
  };

  const confirmIntervention = async () => {
    if (!interventionPreview || isDirecting || isGenerating) return;
    setIsDirecting(true);
    setErrorMessage("");
    try {
      await confirmStoryIntervention(sessionId, {
        intervention: interventionPreview,
        force: interventionPreview.mode === "override",
        expected_revision: snapshot.revision,
      });
      setInterventionDraft("");
      setInterventionPreview(null);
      await syncSnapshot();
    } catch (error) {
      setErrorMessage(error.message || "无法提交剧情干预。");
      await syncSnapshot();
    } finally {
      setIsDirecting(false);
    }
  };

  const cancelIntervention = async (id) => {
    if (isDirecting || isGenerating) return;
    setIsDirecting(true);
    try {
      await cancelStoryIntervention(sessionId, id, snapshot.revision);
      await syncSnapshot();
    } catch (error) {
      setErrorMessage(error.message || "无法取消剧情干预。");
      await syncSnapshot();
    } finally {
      setIsDirecting(false);
    }
  };

  const savePace = async () => {
    if (isDirecting || isGenerating || paceDraft === snapshot.arc_state?.pace) return;
    setIsDirecting(true);
    try {
      await updateStoryPace(sessionId, paceDraft, snapshot.revision);
      await syncSnapshot();
    } catch (error) {
      setErrorMessage(error.message || "无法更新剧情速度。");
      await syncSnapshot();
    } finally {
      setIsDirecting(false);
    }
  };

  const visibleWorldState = Object.entries(snapshot.world_state || scenario.initial_state || {});

  return (
    <main className="simulation-page">
      <header className="simulation-header">
        <div><span className="sim-eyebrow">LIVE SIMULATION</span><h1>{scenario.title || "场景模拟"}</h1></div>
        <div className="sim-status-strip"><span><i className={isGenerating ? "live" : ""} />{isGenerating ? "推演中" : pausedPage ? "已暂停" : currentPage.isEnd ? "已结束" : "等待继续"}</span><span>阶段 · {snapshot.current_phase || "自由互动"}</span><span>回合 · {snapshot.turn_count || 0}/{snapshot.max_turns || "—"}</span></div>
        <button className="icon-button" type="button" onClick={() => setShowQuitModal(true)} aria-label="退出模拟">×</button>
      </header>

      <section className="mobile-context">
        <details><summary>角色与世界状态</summary><div className="mobile-panels"><Roster agents={agents} onSelect={(agent) => setSelectedCharacterId(agent.id || agent.name)} /><WorldPanel entries={visibleWorldState} scenario={scenario} /></div></details>
      </section>

      <div className="simulation-grid">
        <aside className="sim-sidebar left-panel"><Roster agents={agents} onSelect={(agent) => setSelectedCharacterId(agent.id || agent.name)} /></aside>

        <section className="timeline-panel">
          <div className="timeline-context"><span>第 {currentPage.page || 1} 幕</span><p>{scene || scenario.brief?.premise || prompt}</p></div>
          <div className="timeline-content" ref={contentRef} aria-live="polite">
            {currentPage.messages?.length ? currentPage.messages.map((message, index) => <MessageCard key={`${message.event_id || message.id}-${index}`} message={message} skipToken={skipToken} />) : <div className="timeline-empty"><span className="scene-loader" /><strong>角色正在进入场景</strong><p>第一轮行动会从公开场景和各自掌握的信息开始。</p></div>}
            {currentPage.isEnd ? <div className="ending-card"><strong>本次模拟已收束</strong><p>{currentPage.endReason || snapshot.end_reason || "场景达到自然结束条件。"}</p>{snapshot.winner ? <span>结果：{snapshot.winner}</span> : null}</div> : null}
          </div>

          {errorMessage ? <div className="stream-error" role="alert"><span>{errorMessage}</span><button type="button" onClick={() => setErrorMessage("")}>关闭</button></div> : null}

          <section className="director-console" aria-label="剧情导演台">
            <div className="director-console-heading"><div><span>DIRECTOR</span><h2>干预下一步剧情</h2></div><div className="arc-progress"><span>剧情进度 {progressPercent}%</span><div><i style={{ width: `${progressPercent}%` }} /></div></div></div>
            <div className="pace-control"><label htmlFor="story-pace">推进速度 <strong>{paceDraft <= 20 ? "沉浸" : paceDraft <= 40 ? "舒缓" : paceDraft <= 60 ? "均衡" : paceDraft <= 80 ? "紧凑" : "冲刺"}</strong></label><input id="story-pace" type="range" min="0" max="100" step="10" value={paceDraft} onChange={(event) => setPaceDraft(Number(event.target.value))} onMouseUp={savePace} onTouchEnd={savePace} onKeyUp={savePace} disabled={isGenerating || isDirecting} /><div><span>慢 · 多细节</span><span>快 · 早收束</span></div></div>
            <textarea value={interventionDraft} onChange={(event) => { setInterventionDraft(event.target.value); setInterventionPreview(null); }} onFocus={() => setAutoCountdown(null)} maxLength={3000} placeholder="例如：让暴雨在下一轮切断交通，但不要替任何角色决定是否离开。" disabled={isGenerating || isDirecting} />
            <div className="director-options"><select aria-label="干预作用范围" value={interventionScope} onChange={(event) => { setInterventionScope(event.target.value); setInterventionPreview(null); }} disabled={isGenerating || isDirecting}><option value="next_scene">只影响下一步</option><option value="turns">持续若干轮</option><option value="persistent">持续到取消</option></select>{interventionScope === "turns" ? <input aria-label="持续轮数" type="number" min="1" max="100" value={interventionDuration} onChange={(event) => { setInterventionDuration(Number(event.target.value)); setInterventionPreview(null); }} /> : null}<button type="button" onClick={previewIntervention} disabled={!interventionDraft.trim() || isGenerating || isDirecting}>{isDirecting ? "分析中…" : "预检干预"}</button></div>
            {interventionPreview ? <div className={`intervention-preview ${interventionPreview.has_blocking_conflicts ? "has-conflict" : ""}`}><header><span>{interventionPreview.mode === "guidance" ? "柔性引导" : interventionPreview.mode === "event" ? "事件注入" : "强制改写"}</span><strong>{interventionPreview.normalized_directive}</strong></header>{interventionPreview.event_narration ? <p>{interventionPreview.event_narration}</p> : null}{interventionPreview.conflicts?.map((conflict, index) => <div className={`conflict-row ${conflict.severity}`} key={`${conflict.kind}-${index}`}><b>{conflict.severity === "blocking" ? "冲突" : "注意"}</b><span>{conflict.message}</span></div>)}<footer><button type="button" onClick={() => setInterventionPreview(null)}>返回修改</button><button type="button" className="confirm-intervention" onClick={confirmIntervention} disabled={interventionPreview.has_blocking_conflicts && interventionPreview.mode !== "override"}>{interventionPreview.mode === "override" ? "确认强制改写" : "确认应用"}</button></footer></div> : null}
            {pendingInterventions.length ? <div className="pending-interventions"><span>等待/持续生效</span>{pendingInterventions.map((item) => <div key={item.id}><p><b>{item.mode === "guidance" ? "引导" : "事件"}</b>{item.normalized_directive}</p><button type="button" onClick={() => cancelIntervention(item.id)} disabled={isGenerating || isDirecting}>取消</button></div>)}</div> : null}
            {autoCountdown !== null ? <small className="auto-countdown">{autoCountdown} 秒后自动继续；输入干预会立即暂停倒计时。</small> : null}
          </section>

          <footer className="simulation-controls">
            <div className="page-navigation"><button type="button" onClick={() => setCurrentPageIndex((value) => Math.max(0, value - 1))} disabled={currentPageIndex === 0 || isGenerating}>←</button><span>{currentPageIndex + 1} / {pages.length || 1}</span><button type="button" onClick={nextPage} disabled={isGenerating || currentPage.isEnd}>→</button></div>
            <div className="control-actions">
              <button type="button" className="export-control" onClick={exportStory} disabled={isExporting} title="包含完整人物设定、角色秘密、运行状态和全部历史">{isExporting ? "正在导出…" : "导出完整档案"}</button>
              <button type="button" onClick={() => setSkipToken((value) => value + 1)}>跳过打字</button>
              {isGenerating ? <button type="button" className="pause-button" onClick={pauseGeneration}>暂停生成</button> : pausedPage ? <button type="button" className="primary-control" onClick={() => generatePage(pausedPage.index, pausedPage.page)}>继续生成</button> : !currentPage.isEnd ? <button type="button" className="primary-control" onClick={nextPage}>继续推演</button> : null}
            </div>
          </footer>
        </section>

        <aside className="sim-sidebar right-panel">
          <WorldPanel entries={visibleWorldState} scenario={scenario} />
          <section className="control-panel"><h2>推演控制</h2><label>每幕显示条数<select value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} disabled={isGenerating}><option value="3">3 · 短幕</option><option value="6">6 · 标准幕</option><option value="10">10 · 长幕</option></select></label><label className="toggle-row"><span>自动继续</span><input type="checkbox" checked={autoAdvance} onChange={(event) => setAutoAdvance(event.target.checked)} /></label><small>显示条数只控制分页；剧情推进速度由导演台单独控制。</small></section>
        </aside>
      </div>

      {showQuitModal ? <div className="quit-modal-overlay" onClick={() => setShowQuitModal(false)}><div className="quit-modal" role="dialog" aria-modal="true" aria-labelledby="quit-title" onClick={(event) => event.stopPropagation()}><h2 id="quit-title">保存并退出？</h2><p>已经完成的推演会保存在本机历史中，之后可以继续。</p><div><button type="button" onClick={() => setShowQuitModal(false)}>继续留在这里</button><button type="button" className="primary-exit" onClick={confirmQuit}>保存并退出</button></div></div></div> : null}
      {selectedCharacter ? <div className="character-modal-overlay" onClick={() => setSelectedCharacterId("")}><section className="character-modal" role="dialog" aria-modal="true" aria-labelledby="character-modal-title" onClick={(event) => event.stopPropagation()} style={{ "--role-color": roleColor(selectedCharacter.name) }}><button type="button" className="character-modal-close" onClick={() => setSelectedCharacterId("")} aria-label="关闭人物详情">×</button><div className="character-modal-avatar">{selectedCharacter.name?.slice(0, 1)}</div><span className="character-modal-eyebrow">PUBLIC CHARACTER PROFILE</span><h2 id="character-modal-title">{selectedCharacter.name}</h2><strong>{selectedCharacter.public_identity || selectedCharacter.public_profile || "参与者"}</strong><dl><div><dt>公开背景</dt><dd>{selectedCharacter.public_background || "暂无额外公开背景。"}</dd></div><div><dt>形象与特征</dt><dd>{selectedCharacter.public_traits || "暂无额外公开特征。"}</dd></div><div><dt>当前位置</dt><dd>{selectedCharacter.current_location || selectedCharacter.initial_location || "场景中"}</dd></div><div><dt>当前状态</dt><dd>{selectedCharacter.alive === false ? "已离场" : selectedCharacter.active === false ? "暂不可行动" : "在场并可行动"}</dd></div></dl>{selectedCharacter.has_private_context ? <p className="private-context-note">此人物还有仅供自身与导演使用的私密动机，公开详情不会泄露这些内容。</p> : null}</section></div> : null}
    </main>
  );
}

function Roster({ agents, onSelect }) {
  return <section className="roster-panel"><div className="panel-heading"><span>CAST</span><h2>角色阵容</h2></div><div className="roster-list">{agents.map((agent) => <button type="button" className={`roster-card ${!agent.active || agent.alive === false ? "inactive" : ""}`} key={agent.id || agent.name} style={{ "--role-color": roleColor(agent.name) }} onClick={() => onSelect(agent)}><div className="roster-avatar">{agent.name?.slice(0, 1)}</div><div><strong>{agent.name}</strong><small>{agent.public_identity || agent.public_profile || "参与者"}</small><span>{agent.alive === false ? "已离场" : agent.current_location || agent.initial_location || "场景中"}</span></div><b aria-hidden="true">›</b></button>)}</div></section>;
}

function WorldPanel({ entries, scenario }) {
  return <section className="world-panel"><div className="panel-heading"><span>WORLD STATE</span><h2>公开状态</h2></div><dl>{entries.length ? entries.map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{typeof value === "object" ? JSON.stringify(value) : String(value)}</dd></div>) : <p>暂无公开状态变化。</p>}</dl>{scenario.locations?.length ? <><h3>场景地点</h3><div className="location-tags">{scenario.locations.map((location) => <span key={location}>{location}</span>)}</div></> : null}</section>;
}

export default StoryPage;
