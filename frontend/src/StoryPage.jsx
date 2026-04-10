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

  const savedPages = JSON.parse(localStorage.getItem("story_pages") || "[]");
  const savedIndex = localStorage.getItem("current_page_index");

  const [pages, setPages] = useState(savedPages);
  const [currentPageIndex, setCurrentPageIndex] = useState(
    savedPages.length === 0 ? 0 : savedIndex ? Number(savedIndex) : 0
  );
  const [displayedMessages, setDisplayedMessages] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTypingPage, setIsTypingPage] = useState(false);
  const [showQuitModal, setShowQuitModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const contentRef = useRef(null);
  const typingTokenRef = useRef(0);
  const pageDoneRef = useRef(false);

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
    if (!userPrompt || !sessionId) {
      navigate("/");
      return;
    }
  }, [userPrompt, sessionId, navigate]);

  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [displayedMessages, isGenerating]);

  useEffect(() => {
    if (!sessionId || isGenerating) return;

    // 没有任何页面时，自动生成第一页
    if (pages.length === 0) {
      handleGenerateStreamPage(0, 1);
      return;
    }

    // 已存在页面时，切换到旧页直接显示完整内容
    if (currentPage?.messages?.length) {
      showFullPageImmediately(currentPage.messages);
    }
  }, [currentPageIndex, pages.length, sessionId]);

  const openErrorModal = (message) => {
    setErrorMessage(message);
  };

  const closeErrorModal = () => {
    setErrorMessage("");
  };

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

  const randomBreathDelay = () => {
    return 200 + Math.floor(Math.random() * 201); // 200 ~ 400ms
  };

  const createTypingPlaceholder = (msg, indexInPage) => {
    return {
      ...msg,
      side: getAlternatingSide(indexInPage),
      visibleText: "",
      isTyping: true,
      isPlaceholder: true,
    };
  };

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
      visibleText: msg.display_text || "",
      isTyping: false,
    }));

    setDisplayedMessages(fullMessages);
    setIsTypingPage(false);
  };

  const appendTypingMessage = async (msg, indexInPage, pageIndex) => {
      const token = typingTokenRef.current;
      const side = getAlternatingSide(indexInPage);
      const text = msg.display_text || "";

      // 1. 先显示“正在输入”的占位气泡
      setDisplayedMessages((prev) => [
        ...prev,
        createTypingPlaceholder(msg, indexInPage),
      ]);

      // 2. 让占位气泡先存在一下，制造“对方正在输入”的感觉
      await wait(350);

      if (typingTokenRef.current !== token) return;

      // 3. 把占位气泡切换成真正消息，但先从空文本开始
      setDisplayedMessages((prev) =>
        prev.map((item) =>
          item.id === msg.id
            ? {
                ...item,
                side,
                visibleText: "",
                isTyping: true,
                isPlaceholder: false,
              }
            : item
        )
      );

      const speed = getRoleSpeed(msg.speaker);

      // 4. 逐字显示
      await typeText(
        text,
        speed,
        (partial) => {
          setDisplayedMessages((prev) =>
            prev.map((item) =>
              item.id === msg.id
                ? {
                    ...item,
                    visibleText: partial,
                    isTyping: true,
                    isPlaceholder: false,
                  }
                : item
            )
          );
        },
        token
      );

      if (typingTokenRef.current !== token) return;

      // 5. 打完后先保留光标闪一下，不要立刻停
      setDisplayedMessages((prev) =>
        prev.map((item) =>
          item.id === msg.id
            ? {
                ...item,
                visibleText: text,
                isTyping: true,
                isPlaceholder: false,
              }
            : item
        )
      );

      await wait(450);

      if (typingTokenRef.current !== token) return;

      // 6. 光标停止
      setDisplayedMessages((prev) =>
        prev.map((item) =>
          item.id === msg.id
            ? {
                ...item,
                visibleText: text,
                isTyping: false,
                isPlaceholder: false,
              }
            : item
        )
      );

      // 7. 新句子之间加呼吸停顿
      await wait(randomBreathDelay());

      if (typingTokenRef.current !== token) return;

      setPages((prev) =>
        prev.map((page, idx) => {
          if (idx !== pageIndex) return page;
          return pageDoneRef.current ? { ...page, hasPlayed: true } : page;
        })
      );
    };

  const handleGenerateStreamPage = async (targetPageIndex, targetPageNumber) => {
    if (isGenerating) return;

    setIsGenerating(true);
    setIsTypingPage(true);
    pageDoneRef.current = false;
    typingTokenRef.current = Date.now();

    const emptyPage = {
      page: targetPageNumber,
      isEnd: false,
      hasPlayed: false,
      messages: [],
    };

    setPages((prev) => {
      const updated = [...prev];
      updated[targetPageIndex] = emptyPage;
      return updated;
    });

    setCurrentPageIndex(targetPageIndex);
    setDisplayedMessages([]);

    try {
      const response = await fetch("http://127.0.0.1:5000/api/story/next-stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          batch_size: 10,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`生成页面失败（${response.status}）`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let messageIndex = 0;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          const event = JSON.parse(line);

          if (event.type === "message") {
            const newMsg = event.message;

            setPages((prev) =>
              prev.map((page, idx) =>
                idx === targetPageIndex
                  ? { ...page, messages: [...page.messages, newMsg] }
                  : page
              )
            );

            await appendTypingMessage(newMsg, messageIndex, targetPageIndex);
            messageIndex += 1;
          }

          if (event.type === "page_done") {
            pageDoneRef.current = true;

            setPages((prev) =>
              prev.map((page, idx) =>
                idx === targetPageIndex
                  ? {
                      ...page,
                      isEnd: event.isEnd,
                      hasPlayed: true,
                    }
                  : page
              )
            );
          }
        }
      }
    } catch (error) {
      console.error(error);
      openErrorModal(error.message || "生成页面失败");
    } finally {
      setIsGenerating(false);
      setIsTypingPage(false);
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

    await handleGenerateStreamPage(nextIndex, currentPage.page + 1);
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
            <div
              key={`${currentPage.page}-${msg.id}-${idx}`}
              className={`message-row ${msg.side}`}
            >
              <div className="message-meta">{msg.speaker}</div>

              {msg.isPlaceholder ? (
                <div className="message-bubble typing-placeholder-bubble">
                  <span className="typing-placeholder-dot" />
                  <span className="typing-placeholder-dot" />
                  <span className="typing-placeholder-dot" />
                </div>
              ) : (
                <div className="message-bubble">
                  <span>{msg.visibleText}</span>
                  {msg.isTyping && <span className="typing-caret">|</span>}
                </div>
              )}
            </div>
          ))}

          {isGenerating && displayedMessages.length === 0 && (
            <div className="page-loading-zone">
              <div className="page-loading-text">正在生成内容...</div>
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
              ? "正在书写这一页..."
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

      {errorMessage && (
        <div className="quit-modal-overlay">
          <div className="quit-modal">
            <div className="quit-modal-title">提示</div>
            <div className="quit-modal-text">{errorMessage}</div>
            <div className="quit-modal-actions">
              <button className="quit-modal-btn primary" onClick={closeErrorModal}>
                我知道了
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default StoryPage;