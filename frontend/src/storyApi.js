import { getApiErrorMessage, readApiError } from "./apiErrors.js";

async function readNdjson(response, onEvent) {
  if (!response.body) {
    throw new Error("后端没有返回可读取的数据流。");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let ready = false;

  const deliver = async (event) => {
    await onEvent(event);
    if (event.type === "error") {
      throw new Error(getApiErrorMessage(event, "生成场景失败，请稍后重试。"));
    }
    if (event.type === "story_ready") ready = true;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        await deliver(event);
      }
    }

    if (buffer.trim()) {
      const event = JSON.parse(buffer);
      await deliver(event);
    }
    if (!ready) throw new Error("构建连接中断，可从已保存的检查点继续。");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export async function startStoryBuild({ prompt, scene, buildId, signal, onEvent }) {
  const response = await fetch("/api/story/start-stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, scene, ...(buildId ? { build_id: buildId } : {}) }),
    signal,
  });
  if (!response.ok) {
    throw new Error(await readApiError(response, "启动实验失败，请稍后重试。"));
  }
  await readNdjson(response, onEvent);
}

export async function cancelStoryBuild(buildId) {
  const response = await fetch(`/api/story/build/${encodeURIComponent(buildId)}/cancel`, { method: "POST" });
  if (!response.ok && response.status !== 404) throw new Error(await readApiError(response, "取消构建失败，请重试。"));
}

export async function fetchStoryBuildStatus(buildId) {
  const response = await fetch(`/api/story/build/${encodeURIComponent(buildId)}`);
  if (!response.ok) throw new Error(await readApiError(response, "读取构建检查点失败。"));
  return response.json();
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

async function mutation(response, fallback) {
  if (!response.ok) throw new Error(await readApiError(response, fallback));
  return response.json();
}

export async function previewStoryIntervention(sessionId, payload) {
  return mutation(await fetch(`/api/story/session/${encodeURIComponent(sessionId)}/interventions/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }), "无法预检这条剧情干预。");
}

export async function confirmStoryIntervention(sessionId, payload) {
  return mutation(await fetch(`/api/story/session/${encodeURIComponent(sessionId)}/interventions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }), "无法提交这条剧情干预。");
}

export async function cancelStoryIntervention(sessionId, interventionId, expectedRevision) {
  return mutation(await fetch(`/api/story/session/${encodeURIComponent(sessionId)}/interventions/${encodeURIComponent(interventionId)}`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_revision: expectedRevision }),
  }), "无法取消这条剧情干预。");
}

export async function updateStoryPace(sessionId, pace, expectedRevision) {
  return mutation(await fetch(`/api/story/session/${encodeURIComponent(sessionId)}/pace`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pace, expected_revision: expectedRevision }),
  }), "无法更新剧情推进速度。");
}
