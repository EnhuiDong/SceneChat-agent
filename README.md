# SceneChat-Agent

SceneChat-Agent 是一个共享大模型、角色上下文隔离的社会情境模拟原型。模型客户端可以复用，但每个角色拥有独立的完整档案、目标、关系认知、私人记忆和可见观察，从而避免角色获得不应知道的“上帝视角”信息。

世界观和角色生成 Prompt 采用题材忠实原则：“社会实验”是互动观察方法，而不是固定的未来 AI 题材。桌游、电竞、日常生活、历史、奇幻等请求会保持各自的类型、人数、规则与叙事尺度。

## 核心流程

1. 先把用户原文整理成结构化 `ScenarioBrief` 约束账本，区分短输入、部分设定与详细设定，并保留人数、人物、规则、固定事实、目标剧情节点和信息可见性。
2. 根据同一份约束账本分别生成 `WorldSpec` 和定长 `CharacterSpec[]`；用户详细设定优先原样落实，短输入只补足可运行所需的信息。
3. 执行确定性校验：开场、角色数量与姓名、目标、决策逻辑、阶段转换、胜负归属和所有锁定约束都必须通过；失败时按配置进行有限次数的定向结构化修复。
4. 直接从结构化角色对象建立 `AgentState`，Markdown 仅用于归档和兼容旧入口，不再承担新流程的数据协议。
5. 公共世界、导演信息、角色/身份/地点事实分别带 `public`、`director_only`、`audience_only`、`agent:*`、`role:*` 或 `location:*` scope。短背景直接按 scope 注入，只有长背景进入临时 Chroma。
6. 调度器根据公共资格、当前阶段和策略选择行动者，再构造不含越权事实的 `AgentView`；同地、异地、离场角色会获得不同观察。
7. 角色模型只提交 `Intent`。`IntentResolver` 校验阶段、能力、次数、目标、地点与 effect 白名单后才生成权威 `StatePatch`；模型提交的任意 patch 不会直接执行。
8. 阶段、位置、资源、技能、关系、目标、投票、淘汰、定向认知和组合结束条件随推演更新。自然结束、连续失败阻断与可配置的安全上限彼此区分。
9. 运行时使用单调递增的 `revision` 和会话操作锁保护并发推进。用户干预先经过模型归一化和冲突预检，再经过代码白名单校验；柔性引导只进入导演上下文，事件注入与明确确认的强制改写才会成为权威时间线事件。
10. 剧情目标会转换为稳定的 `BeatSpec`。独立的 0–100 节奏值会改变旁白频率、停滞触发阈值、同时活跃的目标节点数和软性收束区间；剧情进度由已完成节点权重计算，不与每幕显示条数混用。
11. 每个结构化角色拥有独立 `voice_profile`，记录语域、句长、直接程度、情绪外显、礼貌程度、表达策略和措辞边界。用户明确语言设定优先；缺失内容只按身份、时代和性格克制补全，不依赖重复口癖制造差异。
12. 角色 Intent 会声明 `addressed_to`、`reply_to_event_id`、`thread_id`、`reply_to_obligation_id`、`conversation_move` 和紧迫度。直接问题、请求与挑战进入独立的 `ConversationThread`；“已经回应”和“已经解决”分别记录，调度器优先处理仍开放的义务，不会用对议题 A 的回答错误关闭议题 B。
13. 长期记忆不再保存每轮泛泛的 `private_reason`，而是从结构化候选中保留主张、线索、承诺、关系证据、信息披露和关键决定，并按当前人物与议题筛选。角色还可拥有带来源、可信度以及 `heard/observed/inferred/verified/disproved` 状态的认知记录；他人的发言不会自动升级为世界事实。核心信念是可选的人物驱动力，没有可靠设定依据时保持为空。
14. 被直接提问的角色仍拥有最高回应优先级；被点名但未被直接询问的角色会得到有时效的可选插话机会。关系维度由每个世界按题材声明，不再固定为猜疑类三轴：战斗可使用敌意、威胁或敬重，合作、家庭、职场及其他题材可使用各自适合的维度。每次变化必须引用角色可见的新事件，并按事件类型限幅；同一证据不能重复作用于同一维度。
15. 信息披露压力按议题分别保存，不会因一个问题受压就泄露其他秘密。Intent 提交前会检查自我重复、复述、漏回应、超长独白和不可见秘密；非公开事实可附带不可推断命题以加强改写泄漏检查。一次重试仍失败时，独立的对话策略会按语言画像生成不新增事实、不替他人决策的显式回应。
16. 导演台提供可折叠的运行观察面板，按“调度、议题、关系、认知、质量”分层展示最近选角原因、待回应义务、分议题压力、场景关系维度、认知来源和确定性质量信号。面板可跳回对应时间线事件，并明确区分人物主观认知与世界事实；它不展示模型思维链。运行观察随流式消息更新，桌面端与移动端均控制默认信息密度。

Web 界面按“写下设定 → 真实构建进度 → 公开信息审阅 → 实时模拟”组织。审阅页提供约束覆盖、运行阶段、公开规则和角色卡，但不会返回导演信息或角色秘密；模拟工作台提供可点开的公开人物卡、公开世界状态、导演干预预检卡、剧情进度与节奏滑块、分页条数、暂停/恢复、跳过打字和可选自动推进。用户开始输入或预检干预时，自动推进会暂停并保留倒计时。首页会列出保存在本机的历史推演，可继续、逐个删除或一键清空。

启动实验时先检查向量模型，再通过与正式生成相同的 JSON 路径检查生成模型；两项检查均通过后才生成约束、世界和角色，避免完成部分生成后才发现服务不可用。向量检查使用实际索引所用的批量接口；背景达到 RAG 阈值时才建立向量索引，短背景仍直接注入。

场景校验会核对阶段和规则能否由真实角色执行、纯环境阶段是否有角色行动出口，以及结束条件的类型和状态更新来源。`all_active` 等全体角色选择器按全体可行动角色处理；未知职能和不存在的主持人会触发定向修复。运行中如果角色阶段没有合法行动者，会明确暂停并报告原因，不再用连续旁白填充。用户未指定称呼时，角色使用独立姓名，职业保留在身份字段；用户给出的姓名、编号和代号保持原样。

文本生成统一通过 OpenAI Python Client 的 Chat Completions 接口调用，可使用 OpenAI、DashScope 或其他实现该协议的兼容服务。向量模型独立配置：兼容服务通过项目内的 LlamaIndex `BaseEmbedding` 适配器接入，DashScope 原生 Embedding 保留为兜底。

## 声明式规则与剧情进度

新建场景使用 `world.execution_version=2`；旧存档缺少版本字段时仍按版本 1 执行，不自动改变原规则语义。

- 行动名称与结果分离：会议投票不会默认淘汰人物，治疗不会默认复活，查验也不会默认泄露阵营。规则以 `effect_mode=rule/ability/stack` 明确结果来源；能力次数和资源成本独立校验。同一动作匹配多个规则时选择最高 `priority`，同优先级冲突在生成校验时报告。
- 所有效果先在副本中逐项校验，包含累计资源消耗；任一项无效则不提交本次行动。规则只支持代码声明的操作，不执行生成的脚本或条件表达式。`legacy_tabletop` 是显式可选的传统桌游效果模板，不按场景名称自动套用。
- 目标可以是人物、`locations` 中的地点或 `entities` 中的物品/提案。实体以 `kind=object/proposal` 和 `state` 声明；`set_entity` 修改已声明字段。`record_vote` 记录投票，`settle_votes` 在全员完成投票后按声明票数阈值更新提案布尔状态；`entity_equals` 可用于结束条件。
- `audience_policy=omniscient` 配合 `reveal_policy=allow_reveal` 可允许全知读者镜头；默认保留悬念。读者镜头仍与角色知识隔离。世界字段默认仅由规则裁决更新，只有 `state_schema.mutable_by` 包含 `director` 的字段可被旁白更新。
- 剧情节点分为 `state` 与 `narrative`。前者根据世界状态、实体状态、人物位置或目标状态，以及已提交行动证据自动完成；后者仅在旁白提议完成时核验近期角色事件。承诺不等于履约、线索不等于找到目标，旁白自己的宣布不能作为状态完成证据。
- 叙事核验每次最多检查 3 个候选节点，不进行自动重试；相同事件证据不会重复请求。默认输出上限 700 tokens、等待上限 45 秒，分别由 `simulation.beat_verification_max_tokens` 和 `simulation.beat_verification_timeout_seconds` 配置。失败或证据不足时保留未完成状态。语义核验依然受所用模型能力影响，并非形式化证明。
- 导演台显示已完成节点数、等待原因和完成证据；可跳转到公开证据所在轮次，私密证据不展开原文。开放场景显示当前阶段，不以运行轮数虚构完成百分比；触及安全轮次上限也不代表故事完成。
- 停滞判断纳入有效状态变化、议题解决和关系变化；重复写入相同状态不算推进。提速仍不能绕过人物选择、信息获取与规则结算。既有完成记录随存档保留，干预不会静默改写过去的完成事实。
- 文本生成、行动、旁白和节点核验记录请求用途、耗时及供应商返回的 token 用量，保存于完整导出的 `simulation.model_requests`；没有返回用量时为 `null`，不记为零。这些记录不是完整账单，不包含向量服务全部调用费用。质量指标没有适用样本时显示 `N/A`，重复率按重复事件而非比较次数计算。

## 项目结构

```text
.
├── app.py                         # Flask Web API
├── main.py                        # 命令行入口：生成设定并开始推演
├── config.json                    # 超时、重试、生成预算等运行与服务参数
├── history.py                     # 命令行模拟入口
├── World.py                       # 世界观生成 Prompt
├── Character.py                   # 角色生成 Prompt
├── scenechat/
│   ├── character_parser.py        # Markdown 角色档案解析
│   ├── config.py                  # config.json 读取与范围校验
│   ├── context.py                 # AgentView 与导演上下文
│   ├── dialogue_quality.py        # 运行时对话质量门与秘密泄漏检查
│   ├── dialogue_policy.py         # 场景中立的回应策略与安全降级
│   ├── embeddings.py              # OpenAI 兼容向量的 LlamaIndex 适配器
│   ├── errors.py                  # 稳定错误码与对外错误信息
│   ├── evaluation.py              # 场景、对话响应/记忆/关系、导演干预与节奏指标
│   ├── generation.py              # 约束账本、世界、角色的分阶段生成
│   ├── knowledge.py               # 实验级隔离索引与角色过滤检索
│   ├── observability.py           # 导演运行观察与确定性质量信号
│   ├── interventions.py           # 导演干预解析、冲突校验与安全应用
│   ├── models.py                  # AgentState / Message / SimulationState
│   ├── openai_compat.py           # OpenAI 兼容传输和 Chat 模型接口
│   ├── pacing.py                   # 节奏策略、剧情节点与进度计算
│   ├── preflight.py               # 模型与向量服务启动前探测
│   ├── providers.py               # 模型与 Embedding 提供商
│   ├── runtime.py                 # Intent / Resolver / StatePatch
│   ├── scenario.py                # 结构化设定模型、渲染与确定性校验
│   ├── scheduler.py               # 多策略公共状态调度
│   ├── simulation.py              # 角色上下文构造和单轮推演
│   ├── storage.py                 # 实验文档归档
│   └── visibility.py              # scope 规范化与权限判定
├── frontend/                      # React + Vite 前端
└── data/experiments/              # 本地生成的实验档案（Git 忽略）
```

## 环境配置

需要 Python 3.10+ 和 Node.js。模型供应商、模型名称、API 地址和 API Key 从 `.env` 读取；超时、重试次数、生成预算、批次大小、JSON 模式、thinking 开关、数据库路径和 CORS 等运行参数保存在仓库根目录的 `config.json`。修改配置后重启后端生效。

```json
{
  "scenario": {
    "json_repair_retries": 2,
    "semantic_repair_retries": 2,
    "transport_retries": 1,
    "request_timeout_seconds": 120,
    "step_timeout_seconds": 240,
    "build_timeout_seconds": 600,
    "max_requests_per_step": 4,
    "repair_max_tokens": 4096
  },
  "simulation": {
    "intent_max_tokens": 900,
    "narration_max_tokens": 480,
    "max_turns": 120
  }
}
```

复制 `.env.example` 为 `.env`，填写模型连接配置（本地 `.env` 不提交到仓库）：

```dotenv
LLM_PROVIDER=dashscope
LLM_API_KEY=your_api_key
LLM_API_BASE=
LLM_MODEL=qwen-plus

EMBEDDING_PROVIDER=dashscope
EMBEDDING_API_KEY=your_api_key
EMBEDDING_API_BASE=
EMBEDDING_MODEL=text-embedding-v2
```

文本模型与向量模型可以在 `.env` 中使用不同的供应商、地址和模型。`*_API_BASE` 留空时，`openai` 使用 OpenAI 默认地址，`dashscope` 文本生成与 `dashscope_compatible` 向量生成使用 DashScope 默认兼容地址；`openai_compatible` 必须填写对应的 `*_API_BASE`。`MODEL_PROVIDER` 保留为 `LLM_PROVIDER` 的旧别名。`config.json` 不再读取 `provider`、`model`、`api_base` 字段；旧 `.env` 中的超时、重试次数和 Token 预算也不覆盖 `config.json`。

| 配置值 | 文本生成 | 向量生成 |
| --- | --- | --- |
| `dashscope` | OpenAI 兼容接口，并支持 `llm.enable_thinking` | DashScope 原生 LlamaIndex 适配器，批次不超过 20 |
| `openai` | OpenAI 默认或自定义地址 | OpenAI `/embeddings` |
| `openai_compatible` | 自定义兼容地址 | 自定义兼容 `/embeddings` 地址 |
| `dashscope_compatible` | — | 用 OpenAI Client 调用 DashScope 兼容地址 |

`llm.json_mode` 为 `auto` 时，OpenAI 与 DashScope 使用原生 JSON Mode，未知兼容服务只依赖严格 Prompt 和本地 JSON 校验；确认服务支持 `response_format` 后可改为 `native`。结构化长文本默认关闭 DashScope thinking，避免推理 token 占满输出预算而截断 JSON。

`scenario.json_repair_retries` 控制不完整 JSON 的重新生成次数，`scenario.semantic_repair_retries` 控制设定未通过确定性校验时的定向修复次数，`scenario.transport_retries` 只在超时、断连或网关临时故障时重发当前步骤；三者有效范围均为 0–2。鉴权、额度、模型名称和请求参数错误不会重复发送。`simulation.intent_max_tokens` 与 `simulation.narration_max_tokens` 分别控制单轮角色 Intent 和旁白 JSON 的输出预算。

### 构建时间预算与断点恢复

- 自动修复返回有限的 JSON 字段补丁，不再要求重写完整世界和全部角色。补丁禁止修改约束账本、人数和角色 ID；合并后仍须通过完整场景校验。默认修复输出预算为 4096 tokens。
- 明确的协议字段错位会先在本地纠正，例如目标阶段误填在 `set_phase.target` 且 `value` 为空。不会替用户创造规则、覆盖冲突值或升级旧场景执行版本。模型补丁只有减少校验问题且不引入新问题才保存；连续两次无进展会停止，语义修复累计上限由 `scenario.semantic_total_attempts` 控制（默认 6 次，跨检查点继续累计）。重复无效 JSON 也会停止，避免从检查点无限重复付费请求。
- 构建期间禁用 SDK 自动重试，由结构化生成层统一控制传输重试。默认单次文本请求上限 120 秒（同时遵守更短的 `llm.request_timeout_seconds`）、每阶段 240 秒、一次构建尝试共 600 秒。每阶段最多 4 次结构化请求，JSON 与语义修复共享该阶段的时间和请求预算；不会通过多层重试无限延长。
- Web 构建流实时报告阶段、耗时、剩余预算、请求次数和修复问题。修复问题可能涉及隐藏设定，默认折叠显示；不返回完整私密检查点或供应商原始错误。
- 约束账本、世界、角色和修复中的场景保存在同一个本地 SQLite 数据库中。失败后选择“从检查点继续”；修改输入或选择“重新生成全部”会开始新构建。恢复仍会先检查模型可用性，成功构建的重复请求不会覆盖已有推演。
- 取消按钮会通知后端、停止后续步骤并取消本地正在等待的异步模型请求。供应商已收到的请求是否停止计算或计费，由供应商决定。离开构建页或连接断开也会触发本地取消。
- `POST /api/story/start-stream` 返回 `build_started.build_id`；再次传入相同 `build_id` 和原始输入可恢复。`GET /api/story/build/<id>` 查询状态，`POST /api/story/build/<id>/cancel` 请求取消。同一检查点不允许并发续跑；后端异常退出后，旧运行租约最多约 15 秒失效。
- 检查点包含人物秘密等完整设定，和数据库一样仅供本机保存。“清除全部”同时删除构建检查点。旧的同步 `/api/story/start` 接口保留时间预算保护；检查点恢复与实时进度使用流式接口。

## 运行

后端：

```bash
python -m venv .venv
pip install -r requirements.txt
python app.py
```

前端：

```bash
cd frontend
npm install
npm run dev
```

也可以直接运行命令行版本：

```bash
python main.py
```

## 数据与费用

- 每个新实验使用独立的内存向量集合，不会读取旧 `storage_chroma` 数据。
- 生成的世界观、角色档案和完整结构化 `scenario.json` 保存在 `data/experiments/<experiment_id>/`。
- Web 会话在每个事件提交后、向前端发送前写入本机 SQLite，默认文件为 `data/scenechat.db`；可通过 `config.json` 的 `storage.database_path` 修改位置。后端重启后可从首页继续，不要求用户安装或操作数据库。
- SQLite 文件包含导演信息、人物秘密和私人记忆，内容未额外加密；不要将该文件提交到仓库或放入公开同步目录。项目已默认忽略数据库及 WAL/SHM 文件。
- 推演页可随时导出一份完整 JSON 档案自行保存；SQLite 恢复与 JSON 导出互不替代。
- 单次推演默认安全上限为 120 轮，可通过 `config.json` 的 `simulation.max_turns` 调整（有效范围 1–1000）；每页请求仍会受批次上限和剩余总轮数限制。
- `/api/story/next-stream` 支持稳定 `request_id` 和 `expected_page`；完成请求可安全重放，断流重试会从已经提交的消息继续，不会重复推进世界状态。
- 新页面请求还可携带 `expected_revision`；状态已经变化时返回 `409 state_revision_conflict`。同一会话生成期间会拒绝并发状态操作，旧客户端不传版本号时仍保持兼容。
- `POST /api/story/session/<id>/interventions/preview` 只生成预检结果，不改变剧情；确认、取消和节奏更新接口都接受 `expected_revision`，并与页面生成共享同一会话操作锁。
- 柔性引导支持“下一步”“持续若干轮”和“持续到取消”。事件注入只能执行安全的公共状态/移动更新；涉及既有事实、角色状态或阶段的强制改写必须由用户在冲突卡上二次确认。
- 干预预检同时读取固定事实和等待执行的干预。对同一状态字段给出互斥结果时，代码层会阻止后提交的普通事件静默覆盖；用户可以先取消旧干预，或明确确认强制改写。
- 仅观众可见的镜头统一作为 `audience_only` 事件进入时间线，不写入任何角色观察或公共状态。存在对立阵营且尚未进入揭晓阶段时，自动旁白只读取公共信息，避免读者镜头过早泄露隐藏身份；公开事件会按可见范围自动形成观察，不接受模型额外注入角色知识。
- 节奏滑块是导演软控制：慢速保留更多反应和关系细节，快速提高关键事件密度并更早寻求自然收束，但不会绕过阶段规则、结束条件、固定设定或角色信息边界。
- `scenechat.evaluation` 提供可重复的导演干预生命周期、时间线可追踪性、旁白新鲜度、角色事件占比、调度存活性、剧情推进动量、私密镜头隔离、patch 边界以及慢/快节奏对照指标，便于部署方在自己的模型配置上进行质量回归。
- 普通时间线会公开回应对象、被回复事件和对话动作，以维持可追踪的多人交互；`private_reason`、私人关系判断、`short_term_state` 以及结构化记忆候选不会返回给浏览器，只保存在本机完整会话中。
- 普通浏览器会话 API 只返回公共世界和公开角色卡；用户主动调用导出功能时，下载内容会包含完整导演设定、人物秘密、私人记忆、状态和全部事件历史，请妥善保管。
- 首页的单项删除和“清除全部”会永久删除对应 SQLite 记录；执行前会再次确认。右上角“保存并退出”只退出页面，不删除推演。
- API 默认只接受本机 Vite 前端来源；远程部署时使用 `config.json` 的 `server.cors_origins` 明确填写实际前端来源，不建议配置为 `*`。
- 持久化与导出格式当前为 schema v5；旧存档在恢复时补齐缺省字段。高于当前版本的数据库和导入文件会被拒绝，不会降级覆盖。
- API 错误使用稳定的 `code`、`stage` 和中文 `message`；供应商原始错误与请求 ID 只记录在后端日志中，不直接显示给前端用户。
- 世界观、角色和每轮行动都会调用外部模型，请关注服务商额度和费用。
- 每次点击“预检干预”会额外调用文本模型，遇到可恢复错误可能触发下述有限重试；返回修改但不重新预检不会产生新调用。
- 推演和干预预检使用独立的业务重试，禁用叠加的 SDK 自动重试：`simulation.transport_retries` 默认 1 次，只重试临时网络、限流或服务故障；鉴权、欠费和无效请求不重试。`parse_retries` 与 `quality_retries` 分别控制结构和质量修复，单个生成步骤最多额外修复一次。干预结构无效时也可修复一次，但不会自动确认或执行干预。
- `simulation.operation_timeout_seconds` 默认 180 秒，`max_requests_per_operation` 默认 5 次；一次事件中的角色、旁白与节点核验共享预算，节点核验本身仍不重试。向量检索遵守同一时间预算，单批传输重试由 `embedding.max_retries` 控制。达到上限明确报错，不伪造生成成功。连续安全兜底默认最多 3 次（`consecutive_fallback_limit`），随后暂停；兜底标记保存在完整导出中，普通旁白不会重置连续角色兜底计数。

## 长故事、检查点与评测

- 未解决承诺与有效结构化记忆不再按年龄丢弃；普通旧事件归档并按人物、目标、地点与重要性召回。摘要采用可追踪的摘录，不额外调用模型，不把传言自动升级为事实。已有存档中曾经被删掉的记忆无法追溯恢复。
- `simulation.context_section_bytes` 控制可选背景等单段输入预算；`simulation.input_budget_bytes` 控制最终推演请求的 UTF-8 字节上限（不是供应商 token 数）。必要设定过大时明确报错，不静默截断角色档案或硬规则。提高上限前请核对模型上下文容量。
- 每个状态版本保存完整检查点，默认保留最近 40 个，可用 `storage.checkpoint_retention` 调整。检查点与最新快照在同一事务内写入；完整副本会增加本机磁盘占用。
- 推演页的“检查点与剧情分支”可从真实检查点创建独立推演。原推演保留，删除原推演也不会影响已建立的分支。旧存档只能从现有完整状态开始，不能凭聊天记录回溯任意轮次。
- 首页支持 JSON 导入预览与确认，默认新建会话，不覆盖已有数据；当前文件上限 20 MB。导出包含秘密与私人记忆，请只导入可信来源并妥善保管。
- 时间线“本轮变化”只展示已提交状态的前后差异及来源事件。人物目标、关系、认知、承诺等变化须主动打开导演视角，主观认知不代表世界事实。旧事件没有记录差异时不编造解释。

离线对比两份完整导出，不产生模型费用：

```bash
python -m scenechat.benchmark before.json after.json
python -m scenechat.benchmark --cases
```

固定案例包含战斗、合作、调查三个详细设定，以及生活、探险两个短设定。对比时保持模型与公开配置一致，各版本各运行一次，记录干预的时机与内容；长记忆案例须留出足够间隔，不能把短样本当成长故事验证。

可选语义评审：`python -m scenechat.benchmark before.json after.json --judge`。该选项会把两份故事内容发送给 `.env` 配置的生成模型，最多额外请求一次、输出 2000 tokens、总时限 60 秒，不重试；超出输入预算会直接拒绝。评审匿名打乱 A/B 顺序，逐项给出设定忠实、动机、实质回应、因果与收尾判断，缺失或伪造来源事件的评分不予接受。

报告分开列出结构指标、语义判断、耗时、已知 token 用量及未知用量。应用层调用次数不等于 SDK 底层尝试次数；未取得 usage 的失败调用不算零费用，没有单价时不估算金额。存档记录公开配置白名单、代码/提示版本摘要及调用模型，不记录 API Key 或服务地址。评测报告可能包含剧情秘密，不应提交到公共仓库；少量测试通过不等于所有题材都已稳定。
