export function getApiErrorMessage(payload, fallbackMessage) {
  if (typeof payload?.error === "string" && payload.error.trim()) {
    return payload.error;
  }
  if (
    payload?.error &&
    typeof payload.error.message === "string" &&
    payload.error.message.trim()
  ) {
    return payload.error.message;
  }
  if (typeof payload?.message === "string" && payload.message.trim()) {
    return payload.message;
  }
  return fallbackMessage;
}


export async function readApiError(response, fallbackMessage) {
  try {
    const payload = await response.json();
    return getApiErrorMessage(payload, fallbackMessage);
  } catch {
    return fallbackMessage;
  }
}
