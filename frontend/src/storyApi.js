import { getApiErrorMessage, readApiError } from "./apiErrors";

async function readNdjson(response, onEvent) {
  if (!response.body) {
    throw new Error("后端没有返回可读取的数据流。");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "error") {
        throw new Error(getApiErrorMessage(event, "生成场景失败，请稍后重试。"));
      }
      await onEvent(event);
    }
  }

  if (buffer.trim()) {
    const event = JSON.parse(buffer);
    if (event.type === "error") {
      throw new Error(getApiErrorMessage(event, "生成场景失败，请稍后重试。"));
    }
    await onEvent(event);
  }
}

export async function startStoryBuild({ prompt, scene, signal, onEvent }) {
  const response = await fetch("/api/story/start-stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, scene }),
    signal,
  });
  if (!response.ok) {
    throw new Error(await readApiError(response, "启动实验失败，请稍后重试。"));
  }
  await readNdjson(response, onEvent);
}

export async function fetchStorySession(sessionId) {
  const response = await fetch(`/api/story/session/${encodeURIComponent(sessionId)}`);
  if (!response.ok) {
    throw new Error(await readApiError(response, "同步故事状态失败。"));
  }
  return response.json();
}

export async function listStorySessions() {
  const response = await fetch("/api/story/sessions");
  if (!response.ok) {
    throw new Error(await readApiError(response, "读取历史推演失败。"));
  }
  const payload = await response.json();
  return payload.sessions || [];
}

export async function clearStorySessions() {
  const response = await fetch("/api/story/sessions", { method: "DELETE" });
  if (!response.ok) {
    throw new Error(await readApiError(response, "清除历史推演失败。"));
  }
  return response.json();
}

export async function fetchStoryExport(sessionId) {
  const response = await fetch(`/api/story/session/${encodeURIComponent(sessionId)}/export`);
  if (!response.ok) {
    throw new Error(await readApiError(response, "导出完整档案失败。"));
  }
  const disposition = response.headers.get("content-disposition") || "";
  const match = disposition.match(/filename="?([^";]+)"?/i);
  return {
    blob: await response.blob(),
    filename: match?.[1] || `scenechat-${sessionId}.json`,
  };
}

export async function deleteStorySession(sessionId) {
  if (!sessionId) return;
  const response = await fetch(`/api/story/session/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await readApiError(response, "删除推演失败。"));
  }
  return response.json();
}
