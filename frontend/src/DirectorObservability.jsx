import { useState } from "react";

const THREAD_STATUS = {
  active: "推进中",
  dormant: "暂时搁置",
  resolved: "已解决",
};

const OBLIGATION_STATUS = {
  open: "待回应",
  responded: "已回应·未解决",
  satisfied: "已解决",
  withdrawn: "已撤回",
  expired: "已过期",
};

const MOVE_LABEL = {
  question: "问题",
  request: "请求",
  challenge: "质疑",
};

const BELIEF_STATUS = {
  heard: "听闻",
  observed: "观察",
  inferred: "推断",
  believed: "相信",
  verified: "已核验",
  disputed: "存疑",
  disproved: "已推翻",
};

function percent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.round(Math.max(0, Math.min(numeric, 1)) * 100);
}

function EventButton({ eventId, onJumpToEvent, children }) {
  if (!eventId) return null;
  return <button type="button" className="evidence-link" onClick={() => onJumpToEvent?.(eventId)}>{children || "查看对应事件"}</button>;
}

function SchedulerView({ data, onJumpToEvent }) {
  const scheduler = data.scheduler || {};
  const source = scheduler.source_event;
  const agentStates = data.agents || [];
  return <div className="observer-stack">
    <article className="scheduler-explanation">
      <span>最近一次调度</span>
      <strong>{scheduler.actor_name ? `选择 ${scheduler.actor_name}` : scheduler.kind === "event" ? "处理环境事件" : "等待下一轮"}</strong>
      <p>{scheduler.reason || "尚未形成调度决策。"}</p>
      {scheduler.thread_id ? <small>关联议题 · {scheduler.thread_id}</small> : null}
      {source ? <div className="source-preview"><b>触发事件</b><span>第 {source.turn} 回合 · {source.speaker}</span><p>{source.speech || source.action}</p><EventButton eventId={source.event_id} onJumpToEvent={onJumpToEvent} /></div> : null}
    </article>
    <div className="agent-signal-grid">
      {agentStates.map((agent) => <article key={agent.name}>
        <header><strong>{agent.name}</strong><span>{agent.emotion || "平静"} · {percent(agent.emotion_intensity)}%</span></header>
        <p>{agent.conversation_goal || "依据长期目标自主判断"}</p>
        <footer><span>{agent.active_thread_count || 0} 个活跃议题</span><span>{agent.pending_obligation_count || 0} 项待回应</span></footer>
      </article>)}
    </div>
  </div>;
}

function ThreadsView({ data, onJumpToEvent }) {
  const threads = data.threads || [];
  if (!threads.length) return <p className="observer-empty">出现直接问题、请求或挑战后，这里会按议题分别跟踪。</p>;
  return <div className="thread-card-list">{threads.map((thread) => {
    const openCount = thread.obligations?.filter((item) => item.status === "open").length || 0;
    return <article className={`thread-card ${thread.status}`} key={thread.id}>
      <header><div><span>{THREAD_STATUS[thread.status] || thread.status}</span><strong>{thread.topic}</strong></div><b>{openCount ? `${openCount} 待回应` : "无开放义务"}</b></header>
      <p className="thread-participants">{'参与者 · ' + (thread.participants?.join("、") || "未记录")}</p>
      <div className="thread-meter"><span style={{ width: `${percent(thread.tension)}%` }} /></div>
      <small>议题张力 {percent(thread.tension)}% · 最近活跃于第 {thread.last_active_turn} 回合</small>
      {thread.pressures?.length ? <div className="pressure-row">{thread.pressures.map((item) => <span key={item.agent}>{item.agent} 披露压力 {percent(item.value)}%</span>)}</div> : null}
      {thread.obligations?.length ? <div className="obligation-list">{thread.obligations.map((item) => <div key={item.id}>
        <span className={`obligation-status ${item.status}`}>{OBLIGATION_STATUS[item.status] || item.status}</span>
        <p><b>{item.requester}</b> → <b>{item.target}</b> · {MOVE_LABEL[item.move] || item.move}</p>
        <small>{item.summary}</small>
        <EventButton eventId={item.resolution_event_id || item.source_event_id} onJumpToEvent={onJumpToEvent}>{item.resolution_event_id ? "查看回应" : "查看发起事件"}</EventButton>
      </div>)}</div> : null}
    </article>;
  })}</div>;
}

function RelationshipsView({ data, onJumpToEvent }) {
  const relationships = data.relationships || [];
  if (!relationships.length) return <p className="observer-empty">角色基于新事件调整关系后，这里会展示当前维度与证据。</p>;
  return <div className="relationship-card-list">{relationships.map((relation) => <article key={`${relation.observer}-${relation.target}`}>
    <header><strong>{relation.observer}</strong><span>看待</span><strong>{relation.target}</strong></header>
    {relation.summary ? <p>{relation.summary}</p> : null}
    <div className="facet-list">{relation.facets?.map((facet) => <div key={facet.id}>
      <label><span>{facet.label}</span><b>{percent(facet.value)}%</b></label>
      <div><i style={{ width: `${percent(facet.value)}%` }} /></div>
      <small><span>{facet.low_label}</span><span>{facet.high_label}</span></small>
    </div>)}</div>
    {relation.latest_evidence ? <details className="relationship-evidence"><summary>最近变化依据</summary><p>{relation.latest_evidence.note || "由对应事件触发"}</p><div>{relation.latest_evidence.changes?.map((change) => <span key={change.id}>{change.label} {Number(change.delta) >= 0 ? "+" : ""}{Number(change.delta).toFixed(2)}</span>)}</div><EventButton eventId={relation.latest_evidence.event_id} onJumpToEvent={onJumpToEvent} /></details> : null}
  </article>)}</div>;
}

function BeliefsView({ data, onJumpToEvent }) {
  const agents = data.agents || [];
  if (!agents.length) return <p className="observer-empty">暂无人物认知状态。</p>;
  return <div className="belief-agent-list">{agents.map((agent) => <article key={agent.name}>
    <header><strong>{agent.name}</strong><span>人物认知，不等于世界事实</span></header>
    {agent.core_beliefs?.length ? <div className="core-belief"><b>核心信念（可选）</b>{agent.core_beliefs.map((item) => <p key={item}>{item}</p>)}</div> : <p className="no-core-belief">未设定额外核心信念</p>}
    {agent.recent_beliefs?.length ? <div className="belief-list">{agent.recent_beliefs.map((item) => <div key={item.id}>
      <span className={`belief-status ${item.epistemic_status}`}>{BELIEF_STATUS[item.epistemic_status] || item.epistemic_status}</span>
      <p>{item.content}</p>
      <small>可信度 {percent(item.confidence)}% · 来源 {item.source_agent || "自身判断"}</small>
      <EventButton eventId={item.source_event_id} onJumpToEvent={onJumpToEvent}>查看来源</EventButton>
    </div>)}</div> : <p className="observer-empty compact">还没有需要持续追踪的认知记录。</p>}
  </article>)}</div>;
}

function QualityView({ data }) {
  const signals = data.quality_signals || [];
  if (!signals.length) return <p className="observer-empty">完成一轮推演后，这里会显示可重复计算的运行信号。</p>;
  return <div className="quality-signal-grid">{signals.map((signal) => <article key={signal.id}>
    <header><strong>{signal.label}</strong><b>{percent(signal.value)}%</b></header>
    <div><span style={{ width: `${percent(signal.value)}%` }} /></div>
    <p>{signal.description}</p>
  </article>)}</div>;
}

export default function DirectorObservability({ data = {}, onJumpToEvent }) {
  const [tab, setTab] = useState("scheduler");
  const summary = data.thread_summary || {};
  const tabs = [
    ["scheduler", "调度"],
    ["threads", "议题"],
    ["relationships", "关系"],
    ["beliefs", "认知"],
    ["quality", "质量"],
  ];
  return <details className="director-observer">
    <summary><div><span>RUNTIME OBSERVER</span><strong>导演运行观察</strong></div><p>{summary.active || 0} 个活跃议题 · {summary.open_obligations || 0} 项待回应</p></summary>
    <div className="observer-body">
      <p className="observer-privacy-note">这里展示已提交的结构化状态和证据，不展示模型思维链。认知与关系是人物主观状态，不自动代表世界事实。</p>
      <div className="observer-tabs" role="tablist" aria-label="导演观察分类">{tabs.map(([id, label]) => <button type="button" role="tab" aria-selected={tab === id} className={tab === id ? "active" : ""} onClick={() => setTab(id)} key={id}>{label}</button>)}</div>
      <div className="observer-tab-content" role="tabpanel">
        {tab === "scheduler" ? <SchedulerView data={data} onJumpToEvent={onJumpToEvent} /> : null}
        {tab === "threads" ? <ThreadsView data={data} onJumpToEvent={onJumpToEvent} /> : null}
        {tab === "relationships" ? <RelationshipsView data={data} onJumpToEvent={onJumpToEvent} /> : null}
        {tab === "beliefs" ? <BeliefsView data={data} onJumpToEvent={onJumpToEvent} /> : null}
        {tab === "quality" ? <QualityView data={data} /> : null}
      </div>
    </div>
  </details>;
}
