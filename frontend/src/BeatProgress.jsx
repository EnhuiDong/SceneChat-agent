const STATUS_LABELS = {
  available: "待推进", in_progress: "进行中", completed: "已完成", skipped: "已跳过",
};

export default function BeatProgress({ arc = {}, phase, onJumpToEvent }) {
  const milestones = arc.progress_mode === "milestones";
  const percent = Math.round((arc.progress || 0) * 100);
  return <>
    <div className="director-console-heading">
      <div><span>DIRECTOR</span><h2>干预下一步剧情</h2></div>
      <div className="arc-progress">
        <span>{milestones ? `已完成节点 ${arc.completed_count}/${arc.total_count}` : `开放推进 · ${phase || "当前场景"}`}</span>
        {milestones && <div role="progressbar" aria-label="节点完成度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}><i style={{ width: `${percent}%` }} /></div>}
      </div>
    </div>
    <p className="beat-progress-note">安全轮次上限不代表故事结局。{arc.plan_adjusted_at_turn != null && `第 ${arc.plan_adjusted_at_turn} 轮收到干预；既有完成记录保留。`}</p>
    {milestones && <details className="beat-evidence">
      <summary>查看节点与完成证据（可能包含剧透）</summary>
      {(arc.beats || []).map((beat) => <article key={beat.id}>
        <strong>{beat.description}</strong>
        <p>{STATUS_LABELS[beat.status] || "待推进"}{beat.blocked_reason && ` · ${beat.blocked_reason}`}</p>
        {!beat.completion && beat.last_check && <p>最近核验：{beat.last_check.reason}</p>}
        {beat.completion && <>
          <p>第 {beat.completion.completed_at_turn} 轮 · {beat.completion.reason || "旧版记录，未核验历史证据"}</p>
          {(beat.evidence || []).map((event) => event.visible
            ? <button type="button" key={event.id} onClick={() => onJumpToEvent?.(event.id)}>第 {event.turn} 轮 · {event.speaker}：{event.excerpt}</button>
            : <span key={event.id}>第 {event.turn} 轮 · 私密行动证据</span>)}
          {!(beat.completion.evidence_event_ids || []).length && <p>旧存档未记录证据。</p>}
        </>}
      </article>)}
    </details>}
  </>;
}
