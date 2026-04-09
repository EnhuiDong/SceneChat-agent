import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./StoryPage.css";

function StoryPage() {
  const location = useLocation();
  const navigate = useNavigate();

  const userPrompt =
    location.state?.prompt || localStorage.getItem("story_prompt") || "";

  const scene =
    location.state?.scene || localStorage.getItem("story_scene") || "";

  const sessionId =
    location.state?.sessionId || localStorage.getItem("story_session_id") || "";

  const savedPages = JSON.parse(localStorage.getItem("story_pages") || "null");
  const savedIndex = localStorage.getItem("current_page_index");

  const [pages, setPages] = useState(savedPages || []);
  const [currentPageIndex, setCurrentPageIndex] = useState(
    savedIndex ? Number(savedIndex) : 0
  );
  const [displayedMessages, setDisplayedMessages] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTypingPage, setIsTypingPage] = useState(false);
  const [showQuitModal, setShowQuitModal] = useState(false);

  const contentRef = useRef(null);
  const typingTokenRef = useRef(0);

  const currentPage = useMemo(() => {
    return pages[currentPageIndex] || { page: 1, isEnd: false, messages: [] };
  }, [pages, currentPageIndex]);

  useEffect(() => {
    localStorage.setItem("story_pages", JSON.stringify(pages));
  }, [pages]);

  useEffect(() => {
    localStorage.setItem("current_page_index", String(currentPageIndex));
  }, [currentPageIndex]);

  useEffect(() => {
    if (!userPrompt || !sessionId || !pages.length) {
      navigate("/");
      return;
    }
  }, [userPrompt, sessionId, pages.length, navigate]);

  useEffect(() => {
    if (!currentPage?.messages) return;

    if (currentPage.hasPlayed) {
      showFullPageImmediately(currentPage.messages);
    } else {
      playTypingForPage(currentPage.messages);
    }
  }, [currentPageIndex, currentPage]);

  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [displayedMessages, isGenerating]);

  const clearStoryCache = () => {
    localStorage.removeItem("story_prompt");
    localStorage.removeItem("story_scene");
    localStorage.removeItem("story_batch_size");
    localStorage.removeItem("story_session_id");
    localStorage.removeItem("story_pages");
    localStorage.removeItem("current_page_index");
  };

  const getAlternatingSide = (index) => {
    return index % 2 === 0 ? "left" : "right";
  };

  const openQuitModal = () => {
    setShowQuitModal(true);
  };

  const closeQuitModal = () => {
    setShowQuitModal(false);
  };

  const confirmQuit = () => {
    clearStoryCache();
    setShowQuitModal(false);
    navigate("/");
  };

  const getRoleSpeed = (speaker) => {
    const speedMap = {
      广播: 22,
      系统: 20,
      旁白: 24,
    };

    if (speedMap[speaker]) return speedMap[speaker];

    const hash = [...speaker].reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
    return 28 + (hash % 18);
  };

  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const typeText = async (fullText, speed, onUpdate, token) => {
    let current = "";
    for (let i = 0; i < fullText.length; i++) {
      if (typingTokenRef.current !== token) return;

      current += fullText[i];
      onUpdate(current);

      const char = fullText[i];
      let delay = speed;

      if ("，。！？；：,.!?".includes(char)) {
        delay += 120;
      } else if ("…".includes(char)) {
        delay += 180;
      }

      await wait(delay);
    }
  };

  const showFullPageImmediately = (messages) => {
    const fullMessages = messages.map((msg, idx) => ({
      ...msg,
      side: getAlternatingSide(idx),
      visibleText: msg.text || "",
      isTyping: false,
    }));

    setDisplayedMessages(fullMessages);
    setIsTypingPage(false);
  };

  const playTypingForPage = async (messages) => {
    const token = Date.now();
    typingTokenRef.current = token;

    setIsTypingPage(true);
    setDisplayedMessages([]);

    for (let i = 0; i < messages.length; i++) {
      if (typingTokenRef.current !== token) return;

      const msg = messages[i];
      const text = msg.text || "";
      const side = getAlternatingSide(i);

      const draft = {
        ...msg,
        side,
        visibleText: "",
        isTyping: true,
      };

      setDisplayedMessages((prev) => [...prev, draft]);

      await wait(500);

      const speed = getRoleSpeed(msg.speaker);

      await typeText(
        text,
        speed,
        (partial) => {
          setDisplayedMessages((prev) =>
            prev.map((item, idx) =>
              idx === i
                ? {
                    ...item,
                    visibleText: partial,
                  }
                : item
            )
          );
        },
        token
      );

      if (typingTokenRef.current !== token) return;

      setDisplayedMessages((prev) =>
        prev.map((item, idx) =>
          idx === i
            ? {
                ...item,
                visibleText: text,
                isTyping: false,
              }
            : item
        )
      );

      await wait(250);
    }

    if (typingTokenRef.current === token) {
      setIsTypingPage(false);

      setPages((prev) =>
        prev.map((page, idx) =>
          idx === currentPageIndex
            ? {
                ...page,
                hasPlayed: true,
              }
            : page
        )
      );
    }
  };

  const handlePrevPage = () => {
    if (currentPageIndex === 0 || isGenerating || isTypingPage) return;
    typingTokenRef.current += 1;
    setCurrentPageIndex((prev) => prev - 1);
  };

  const handleNextPage = async () => {
    if (isGenerating || isTypingPage) return;

    if (currentPage.isEnd) {
      clearStoryCache();
      navigate("/");
      return;
    }

    const nextIndex = currentPageIndex + 1;

    if (pages[nextIndex]) {
      typingTokenRef.current += 1;
      setCurrentPageIndex(nextIndex);
      return;
    }

    setIsGenerating(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/api/story/next", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          batch_size: 5,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        alert(data.error || "生成下一页失败");
        setIsGenerating(false);
        return;
      }

      const messages = data.messages || [];

      const newPage = {
        page: data.page,
        isEnd: data.isEnd,
        hasPlayed: false,
        messages: messages,
      };

      setPages((prev) => [...prev, newPage]);
      setCurrentPageIndex(nextIndex);
    } catch (error) {
      console.error(error);
      alert("生成下一页失败");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="story-page">
      <div className="book-shell">
        <button className="quit-btn" onClick={openQuitModal} title="退出故事">
          ×
        </button>

        <div className="book-header">
          <div className="book-title">社会模拟实验</div>
          <div className="book-subtitle">
            第 {currentPage.page} 页
            {scene ? ` ｜ 场景：${scene}` : ""}
          </div>
          {userPrompt ? <div className="book-prompt">设定：{userPrompt}</div> : null}
        </div>

        <div className="book-content" ref={contentRef}>
          {displayedMessages.map((msg, idx) => (
            <div key={`${currentPage.page}-${msg.id}-${idx}`} className={`message-row ${msg.side}`}>
              <div className="message-meta">{msg.speaker}</div>
              <div className="message-bubble">
                <span>{msg.visibleText}</span>
                {msg.isTyping && <span className="typing-caret">|</span>}
              </div>
            </div>
          ))}

          {isGenerating && (
            <div className="page-loading-zone">
              <div className="page-loading-text">正在生成下一页...</div>
              <div className="page-loading-dots">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
        </div>

        <div className="book-footer">
          <button
            className={`nav-btn left-btn ${currentPageIndex === 0 ? "hidden-btn" : ""}`}
            onClick={handlePrevPage}
            disabled={currentPageIndex === 0 || isGenerating || isTypingPage}
          >
            ←
          </button>

          <div className="page-indicator">
            {isGenerating
              ? "正在书写下一页..."
              : isTypingPage
              ? "文字显现中..."
              : currentPage.isEnd
              ? "故事已结束"
              : "继续阅读"}
          </div>

          <button
            className="nav-btn right-btn"
            onClick={handleNextPage}
            disabled={isGenerating || isTypingPage}
          >
            {currentPage.isEnd ? "End" : "→"}
          </button>
        </div>
      </div>

      {showQuitModal && (
        <div className="quit-modal-overlay">
          <div className="quit-modal">
            <div className="quit-modal-title">结束当前故事？</div>
            <div className="quit-modal-text">
              退出后将返回首页，当前阅读进度会被清除。
            </div>
            <div className="quit-modal-actions">
              <button className="quit-modal-btn secondary" onClick={closeQuitModal}>
                取消
              </button>
              <button className="quit-modal-btn primary" onClick={confirmQuit}>
                确定退出
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default StoryPage;