export function saveStorySetup(storage, prompt, ready) {
  storage.setItem("story_prompt", prompt);
  storage.setItem("story_scene", ready.scene || "");
  storage.setItem("story_batch_size", "6");
  storage.setItem("story_session_id", ready.session_id);
  storage.setItem("story_scenario", JSON.stringify(ready.scenario || {}));
  storage.setItem("story_worldview", ready.worldview || "");
  storage.setItem("story_public_characters", ready.characters || "");
  storage.setItem("story_pages", JSON.stringify([]));
  storage.setItem("current_page_index", "0");
  storage.removeItem("story_pending_page_request");
}

export function loadStoredScenario(storage) {
  try {
    return JSON.parse(storage.getItem("story_scenario") || "{}");
  } catch {
    return {};
  }
}

export function clearStoryStorage(storage, { keepDraft = false } = {}) {
  const draftPrompt = keepDraft ? storage.getItem("story_prompt") || "" : "";
  const draftScene = keepDraft ? storage.getItem("story_scene") || "" : "";
  [
    "story_prompt",
    "story_scene",
    "story_batch_size",
    "story_session_id",
    "story_scenario",
    "story_worldview",
    "story_public_characters",
    "story_pages",
    "current_page_index",
    "story_pending_page_request",
  ].forEach((key) => storage.removeItem(key));
  if (keepDraft) {
    storage.setItem("story_draft_prompt", draftPrompt);
    storage.setItem("story_draft_scene", draftScene);
  }
}
