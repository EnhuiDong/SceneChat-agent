import { useMemo, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { clearStoryStorage, loadStoredScenario } from "./scenarioStorage";
import { deleteStorySession } from "./storyApi";
import "./BuildReview.css";

function ItemList({ items, empty = "未设置" }) {
  if (!items?.length) return <p className="empty-copy">{empty}</p>;
  return <ul className="review-list">{items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul>;
}

function characterTraits(value) {
  if (Array.isArray(value)) return value.filter(Boolean);
  if (typeof value !== "string") return [];
  return value.split(/[、，,；;\n]/).map((item) => item.trim()).filter(Boolean);
}

function ReviewPage() {
  const navigate = useNavigate();
  const scenario = useMemo(() => loadStoredScenario(localStorage), []);
  const sessionId = localStorage.getItem("story_session_id") || "";
  const prompt = localStorage.getItem("story_prompt") || "";
  const scene = localStorage.getItem("story_scene") || "";
  const worldview = localStorage.getItem("story_worldview") || "";
  const [isReturning, setIsReturning] = useState(false);

  if (!sessionId || !scenario.title) {
    return <Navigate to="/" replace />;
  }

  const summary = scenario.constraint_summary || {};
  const constraints = scenario.brief?.constraints || [];

  const returnToEdit = async () => {
    if (isReturning) return;
    setIsReturning(true);
    try {
      await deleteStorySession(sessionId);
    } finally {
      clearStoryStorage(localStorage, { keepDraft: true });
      navigate("/", { replace: true, state: { prompt, scene } });
    }
  };

  return (
    <main className="flow-page review-page">
      <section className="flow-card review-card">
        <header className="review-header">
          <div>
            <div className="flow-eyebrow">SCENARIO REVIEW</div>
            <h1>{scenario.title}</h1>
            <p>{scenario.brief?.premise || scene}</p>
          </div>
          <div className="review-score" title="已纳入结构化场景的锁定约束">
            <strong>{summary.covered ?? 0}/{summary.locked ?? 0}</strong>
            <span>锁定约束已覆盖</span>
          </div>
        </header>

        <div className="review-notice">
          <span>✓</span>
          <p><strong>生成完成，可以在开始前核对。</strong>公开信息展示在这里；角色秘密只会提供给对应角色或导演。</p>
        </div>

        <div className="review-grid">
          <section className="review-section constraints-panel">
            <div className="section-title-row"><h2>约束账本</h2><span>{summary.total || 0} 项</span></div>
            {constraints.length ? constraints.map((item) => (
              <article className="constraint-item" key={item.id}>
                <span>{item.category || "设定"}</span>
                <p>{item.content}</p>
                {item.locked ? <b>已锁定</b> : null}
              </article>
            )) : <p className="empty-copy">简短概念没有显式硬约束，系统已补全可运行结构。</p>}
            {summary.protected > 0 ? (
              <div className="protected-note">🔒 另有 {summary.protected} 项受保护约束，内容不会在公开界面显示。</div>
            ) : null}
          </section>

          <section className="review-section">
            <div className="section-title-row"><h2>运行骨架</h2><span>{scenario.scheduler || "自动调度"}</span></div>
            <h3>阶段</h3>
            <div className="phase-chain">{(scenario.phases || []).map((phase) => <span key={phase}>{phase}</span>)}</div>
            <h3>公开规则</h3>
            <ItemList items={scenario.public_rules} empty="此场景按自由互动推进。" />
            <h3>结束条件</h3>
            <ItemList items={scenario.termination_conditions} empty="由自然剧情收束或安全轮次上限结束。" />
          </section>
        </div>

        <section className="review-section character-section">
          <div className="section-title-row"><h2>角色阵容</h2><span>{scenario.characters?.length || 0} 人</span></div>
          <div className="character-card-grid">
            {(scenario.characters || []).map((character, index) => (
              <article className="character-card" key={character.id} style={{ "--character-index": index }}>
                <div className="character-avatar">{character.name?.slice(0, 1)}</div>
                <div><h3>{character.name}</h3><strong>{character.public_identity || "参与者"}</strong></div>
                <p>{character.public_background || "身份背景将在互动中逐步呈现。"}</p>
                {characterTraits(character.public_traits).length ? <div className="trait-row">{characterTraits(character.public_traits).map((trait) => <span key={trait}>{trait}</span>)}</div> : null}
                <footer><span>起点：{character.initial_location || "场景中心"}</span>{character.has_private_context ? <span>🔒 有私密动机</span> : null}</footer>
              </article>
            ))}
          </div>
        </section>

        {(scenario.brief?.assumptions?.length || scenario.brief?.contradictions?.length || scenario.warnings?.length) ? (
          <section className="review-section decisions-section">
            <h2>生成决策与提醒</h2>
            {scenario.brief.assumptions?.length ? <><h3>为保证可运行而补充</h3><ItemList items={scenario.brief.assumptions} /></> : null}
            {scenario.brief.contradictions?.length ? <><h3>输入中的潜在冲突</h3><ItemList items={scenario.brief.contradictions} /></> : null}
            {scenario.warnings?.length ? <><h3>校验提醒</h3><ItemList items={scenario.warnings} /></> : null}
          </section>
        ) : null}

        <details className="worldview-details">
          <summary>查看完整公开世界观</summary>
          <pre>{worldview || "暂无额外世界观说明。"}</pre>
        </details>

        <footer className="review-actions">
          <button type="button" className="secondary-action" onClick={returnToEdit} disabled={isReturning}>← 返回修改</button>
          <button type="button" className="primary-action" onClick={() => navigate("/story")}>开始模拟 <span>→</span></button>
        </footer>
      </section>
    </main>
  );
}

export default ReviewPage;
