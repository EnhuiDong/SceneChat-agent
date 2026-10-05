# 提示词分布与修改入口

SceneChat 不把所有模型任务合并成一个总提示词。设定生成负责忠实地建立运行条件；角色提交行动意图；旁白呈现结果；引擎负责真正的执行和结束判定。

## 建议阅读顺序

| 文件 | 职责 | 如何生效 |
| --- | --- | --- |
| `scenechat/prompts/actor.md` | 角色如何利用已知信息推进未完成事项、避免重复观点、选择真实行动 | 每次角色生成的提示词开头 |
| `scenechat/prompts/narrator.md` | 旁白的权限、证据要求、何时跳过、不能伪造的结果 | 每次旁白生成的提示词开头 |
| `scenechat/prompts/scenario.md` | 详细/简短输入的补全边界、角色策略、机制闭合与节点设计 | 世界和角色结构化生成 |
| `scenechat/prompt_library.py` | 加载上述文件；生成角色工作单和旁白的引擎事实 | 只从本地状态构造上下文，不调用额外模型 |

修改 Markdown 后重启后端；提示词加载有进程内缓存。这里只写模型行为约束，不修改 JSON 字段名称或引擎允许的操作。模型、供应商和密钥仍由 `.env` 配置。

## 其余提示词与拼装位置

| 文件 / 名称 | 内容 |
| --- | --- |
| `scenechat/generation.py` / `BRIEF_SYSTEM_PROMPT` | 从用户输入提取人数、硬约束、公开与私密范围 |
| 同文件 / `WORLD_JSON_INSTRUCTION`、`RUNTIME_GENERATION_GUIDANCE` | 世界的 JSON 字段及运行协议；这是结构约束，不是剧情剧本 |
| 同文件 / `CHARACTER_JSON_INSTRUCTION` | 角色 JSON、动机、可选信念、能力、知识和语言画像 |
| 同文件 / `repair_scenario_package`、`_invoke_json` | 语义补丁修复、JSON 格式修复；携带具体错误与修复历史 |
| `scenechat/simulation.py` / `build_agent_prompt` | actor.md + 有限视角上下文 + 推进工作单 + 合法动作 + Intent JSON |
| 同文件 / `build_narrator_prompt` | narrator.md + 引擎事实 + 可见历史 + 节奏 + Narrator JSON |
| 同文件 / `simulate_next_turn`、`simulate_narration` | 被拒绝后的定向重试提示；不是另一个导演 |
| `scenechat/context.py` | 角色可见信息、记忆、关系、主张等上下文 |
| `scenechat/pacing.py` / `pacing_context` | 快慢节奏指令及当前节点，不授权跳过规则 |
| `scenechat/scheduler.py` | 非模型提示：点名回应、等待时间与防饥饿调度；不能只靠模型自报紧迫度抢占回合 |
| `scenechat/interventions.py` / `INTERVENTION_PROMPT` | 用户干预的分类、冲突与补丁预检；确认前不执行 |
| `scenechat/beat_evidence.py` / `verify_narrative_candidates` | 根据已提交事件核验叙事节点，不允许旁白自证 |
| `scenechat/benchmark.py` / `judge_pair` | 用户主动启用的匿名对比评审，不参与普通推演 |
| `scenechat/preflight.py` | 最小模型可用性探针，不生成故事 |
| `Character.py` / `CHARACTER_SYSTEM_PROMPT` | 角色设计通用原则；结构化入口叠加 scenario.md 与 JSON 协议 |
| `World.py` / `WORLD_SYSTEM_PROMPT` | 旧 Markdown 世界生成接口；当前结构化世界入口不再使用它 |

## 修改时需要保持的边界

- “推进”是回答、验证、选择、行动或结果的变化；不等于制造新名词、新光效、冲突升级或提前宣布结局。
- 同一行动可以在战斗、修理、照护等场景合理重复，必须结合实际状态；不能只重说同一观点。
- 长段落的跨角色复述同样会被检查；真实消耗资源、移动或执行其他有效动作时，不因台词相似而拒绝整项行动。
- 没有新证据时允许保留分歧，不强迫人物坦白、不让人物凭空获得私密信息。
- 自定义规则用准确的 action_type / rule_id 表达；发言中的意图不能代替动作执行。
- V2 规则仍按 priority 选取，rule_id 不能绕过更高优先级规则；旧 V1 同动作的显式出口可以按 ID 选择。
- 听众不等于回应对象；明确点名的 specific 请求不会因动作里“看向大家”而变为全员待办。
- 固定名单中的演员不能由旁白增加；开放场景的背景人物不能变成新的可操控角色。
- 提示词不能替代执行校验。`runtime.py`、`narrative_grounding.py`、`dialogue_quality.py` 分别检查规则执行、部分结果证据和重复/信息边界。

## 验证方法与限制

修改后同时检查：JSON 是否有效；实际动作及结算是否发生；跨多轮是否重复同一观点；是否引入未授权设定；未完成事项是否真正解决。不要以词汇差异、关系字段变化或进度条变化代替因果验证。

自动检查只能覆盖明确的结构、文本相似度和部分结果表达，不能保证理解所有语义等价或隐喻。旧存档里已经生成的错误剧情不会被自动重写；存在矛盾胜负规则的旧运行会明确报错，建议重新生成。

投票回读依据已提交事件计算，并按可见性隔离；它不是给所有场景增加投票机制。自然语言中的票数、离场与身份公开检查仍是有限规则，不是通用语义证明器。一次受控场景通过也不代表开放剧情、复杂战斗和完整设定生成都已得到质量验证。

提示词组织参考 [OpenAI 官方提示工程文档](https://developers.openai.com/api/docs/guides/prompt-engineering)：明确职责、约束、上下文与输出格式；实际效果仍须在所配置的模型上验证，不意味着更换为 OpenAI 模型。
