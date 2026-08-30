export function loadStoryPages(storage) {
  try {
    const value = JSON.parse(storage.getItem("story_pages") || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

export function loadPageIndex(storage, pageCount) {
  const value = Number(storage.getItem("current_page_index"));
  if (!Number.isInteger(value) || value < 0 || value >= pageCount) {
    return 0;
  }
  return value;
}
