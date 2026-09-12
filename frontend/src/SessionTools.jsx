import { useState } from "react";
import { readApiError } from "./apiErrors";

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await readApiError(response, "操作失败，请重试。"));
  return response.json();
}

export function ImportSession({ onImported }) {
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function read(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setBusy(true); setError(""); setPreview(null);
    try {
      if (file.size > 20 * 1024 * 1024) throw new Error("文件不能超过 20 MB。");
      const payload = JSON.parse(await file.text());
      const summary = await api("/api/story/import/preview", payload);
      setPreview({ payload, summary, requestId: crypto.randomUUID() });
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  async function confirm() {
    setBusy(true); setError("");
    try {
      const result = await api("/api/story/import", { payload: preview.payload, request_id: preview.requestId });
      await onImported({ id: result.session_id });
      setPreview(null);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  return <section className="session-tools" aria-label="导入推演">
    <label>导入完整存档 <input type="file" accept=".json,application/json" disabled={busy} onChange={read} /></label>
    {preview && <div className="session-preview"><strong>{preview.summary.title}</strong>
      <p>{preview.summary.characters} 位人物 · {preview.summary.turns} 个事件 · 存档 v{preview.summary.schema_version}</p>
      <p>{preview.summary.warning}</p>
      <button disabled={busy} onClick={confirm}>{busy ? "导入中…" : "确认导入为新推演"}</button>
      <button disabled={busy} onClick={() => setPreview(null)}>取消</button></div>}
    {error && <p role="alert">{error}</p>}
  </section>;
}

export function CheckpointBranches({ sessionId, disabled, onCreated }) {
  const [points, setPoints] = useState(null);
  const [selected, setSelected] = useState("");
  const [requestId, setRequestId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const base = `/api/story/session/${encodeURIComponent(sessionId)}`;
  async function load() {
    setBusy(true); setError("");
    try {
      const data = await api(`${base}/checkpoints`);
      setPoints(data.checkpoints); setSelected("");
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  async function create() {
    setBusy(true); setError("");
    try {
      const result = await api(`${base}/branch`, { revision: Number(selected), request_id: requestId });
      await onCreated(result.session_id);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  return <details className="session-tools"><summary>检查点与剧情分支</summary>
    <p>复制一个已保存的完整状态，原推演不变。过期检查点不提供回溯。</p>
    <button disabled={disabled || busy} onClick={load}>读取可用检查点</button>
    {points && <><label>从哪个节点继续？<select value={selected} disabled={busy || disabled} onChange={e => { setSelected(e.target.value); setRequestId(crypto.randomUUID()); }}>
      <option value="">选择检查点</option>
      {points.map(p => <option key={p.revision} value={p.revision}>事件 {p.turn_count} · 版本 {p.revision}</option>)}
    </select></label><button disabled={selected === "" || busy || disabled} onClick={create}>{busy ? "处理中…" : "创建并进入新分支"}</button></>}
    {error && <p role="alert">{error}</p>}
  </details>;
}

export function ChangeList({ changes = [] }) {
  return changes.length ? <ul>{changes.map((c, i) => <li key={i}><strong>{c.path}</strong>
    <span>{JSON.stringify(c.before) ?? "未设置"} → {JSON.stringify(c.after) ?? "未设置"}</span>
    <small>依据事件：{c.event_id}</small></li>)}</ul> : <p>本事件没有记录到相关状态变化。</p>;
}

export function EventChanges({ sessionId, message }) {
  const [privateChanges, setPrivateChanges] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const changes = message.public_changes || [];
  async function reveal() {
    setBusy(true); setError("");
    try {
      const data = await api(`/api/story/session/${encodeURIComponent(sessionId)}/changes/${encodeURIComponent(message.event_id)}`);
      setPrivateChanges(data.changes);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }
  return <details className="event-changes"><summary>本轮变化 · {changes.length} 项公开状态</summary>
    <ChangeList changes={changes.slice(0, 3)} />
    {changes.length > 3 && <details><summary>展开剩余 {changes.length - 3} 项</summary><ChangeList changes={changes.slice(3)} /></details>}
    <p>导演视角包含人物秘密和主观认知；认知变化不等于世界事实。</p>
    {privateChanges === null ? <button disabled={busy} onClick={reveal}>我接受剧透，查看导演变化</button> : <><button onClick={() => setPrivateChanges(null)}>隐藏私密变化</button><ChangeList changes={privateChanges} /></>}
    {error && <p role="alert">{error}</p>}
  </details>;
}
