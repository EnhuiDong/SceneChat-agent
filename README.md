# SceneChat-Agent

SceneChat-Agent 是一个共享大模型、角色上下文隔离的社会情境模拟原型。模型客户端可以复用，但每个角色拥有独立的完整档案、目标、关系认知、私人记忆和可见观察，从而避免角色获得不应知道的“上帝视角”信息。

世界观和角色生成 Prompt 采用题材忠实原则：“社会实验”是互动观察方法，而不是固定的未来 AI 题材。桌游、电竞、日常生活、历史、奇幻等请求会保持各自的类型、人数、规则与叙事尺度。

## 核心流程

1. 先把用户原文整理成结构化 `ScenarioBrief` 约束账本，区分短输入、部分设定与详细设定，并保留人数、人物、规则、固定事实、目标剧情节点和信息可见性。
2. 根据同一份约束账本分别生成 `WorldSpec` 和定长 `CharacterSpec[]`；用户详细设定优先原样落实，短输入只补足可运行所需的信息。
3. 执行确定性校验：开场、角色数量与姓名、目标、决策逻辑、结束条件和所有锁定约束都必须通过；失败时只允许一次结构化修复。
4. 直接从结构化角色对象建立 `AgentState`，Markdown 仅用于归档和兼容旧入口，不再承担新流程的数据协议。
5. 公共世界、导演信息、角色/身份/地点事实分别带 `public`、`director_only`、`audience_only`、`agent:*`、`role:*` 或 `location:*` scope。短背景直接按 scope 注入，只有长背景进入临时 Chroma。
6. 调度器根据公共资格、当前阶段和策略选择行动者，再构造不含越权事实的 `AgentView`；同地、异地、离场角色会获得不同观察。
7. 角色模型只提交 `Intent`。`IntentResolver` 校验阶段、能力、次数、目标、地点与 effect 白名单后才生成权威 `StatePatch`；模型提交的任意 patch 不会直接执行。
8. 阶段、位置、资源、技能、关系、目标、投票、淘汰、定向认知和组合结束条件随推演更新。自然结束、连续失败阻断与可配置的安全上限彼此区分。
9. 运行时使用单调递增的 `revision` 和会话操作锁保护并发推进。用户干预先经过模型归一化和冲突预检，再经过代码白名单校验；柔性引导只进入导演上下文，事件注入与明确确认的强制改写才会成为权威时间线事件。
10. 剧情目标会转换为稳定的 `BeatSpec`。独立的 0–100 节奏值会改变旁白频率、停滞触发阈值、同时活跃的目标节点数和软性收束区间；剧情进度由已完成节点权重计算，不与每幕显示条数混用。

Web 界面按“写下设定 → 真实构建进度 → 公开信息审阅 → 实时模拟”组织。审阅页提供约束覆盖、运行阶段、公开规则和角色卡，但不会返回导演信息或角色秘密；模拟工作台提供可点开的公开人物卡、公开世界状态、导演干预预检卡、剧情进度与节奏滑块、分页条数、暂停/恢复、跳过打字和可选自动推进。用户开始输入或预检干预时，自动推进会暂停并保留倒计时。首页会列出保存在本机的历史推演，可继续、逐个删除或一键清空。

启动实验前先通过与正式生成相同的 JSON 路径探测生成模型。生成结构化设定后，只有背景长度达到 RAG 阈值时才探测向量模型并建立索引；短场景不会因为 Embedding 不可用而无法运行。

文本生成统一通过 OpenAI Python Client 的 Chat Completions 接口调用，可使用 OpenAI、DashScope 或其他实现该协议的兼容服务。向量模型独立配置：兼容服务通过项目内的 LlamaIndex `BaseEmbedding` 适配器接入，DashScope 原生 Embedding 保留为兜底。

## 项目结构

```text
.
├── app.py                         # Flask Web API
├── main.py                        # 命令行入口：生成设定并开始推演
├── history.py                     # 命令行模拟入口
├── World.py                       # 世界观生成 Prompt
├── Character.py                   # 角色生成 Prompt
├── scenechat/
│   ├── character_parser.py        # Markdown 角色档案解析
│   ├── context.py                 # AgentView 与导演上下文
│   ├── embeddings.py              # OpenAI 兼容向量的 LlamaIndex 适配器
│   ├── errors.py                  # 稳定错误码与对外错误信息
│   ├── evaluation.py              # 场景、轨迹、导演干预与节奏对照指标
│   ├── generation.py              # 约束账本、世界、角色的分阶段生成
│   ├── knowledge.py               # 实验级隔离索引与角色过滤检索
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

需要 Python 3.10+ 和 Node.js。复制 `.env.example` 为 `.env`，然后填写模型配置：

```dotenv
# dashscope | openai | openai_compatible
LLM_PROVIDER=dashscope
LLM_API_KEY=your_api_key
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
LLM_REQUEST_TIMEOUT_SECONDS=180
LLM_MAX_RETRIES=1
LLM_JSON_MODE=auto
LLM_TOKEN_LIMIT_PARAMETER=max_tokens
LLM_ENABLE_THINKING=false
SIMULATION_PARSE_RETRIES=1

# dashscope | dashscope_native | dashscope_compatible | openai | openai_compatible
EMBEDDING_PROVIDER=dashscope
EMBEDDING_API_KEY=your_api_key
EMBEDDING_API_BASE=
EMBEDDING_MODEL=text-embedding-v2
EMBEDDING_BATCH_SIZE=20
EMBEDDING_REQUEST_TIMEOUT_SECONDS=180
EMBEDDING_MAX_RETRIES=1
```

文本模型与向量模型可以使用不同的供应商、地址和 API Key。`openai` 使用 OpenAI 默认地址；`openai_compatible` 必须填写对应的 `*_API_BASE`。`MODEL_PROVIDER` 仍作为旧 `.env` 的兼容别名，但新配置应使用 `LLM_PROVIDER`。

| 配置值 | 文本生成 | 向量生成 |
| --- | --- | --- |
| `dashscope` | OpenAI 兼容接口，并支持 `LLM_ENABLE_THINKING` | DashScope 原生 LlamaIndex 适配器，批次不超过 20 |
| `openai` | OpenAI 默认或自定义地址 | OpenAI `/embeddings` |
| `openai_compatible` | 自定义兼容地址 | 自定义兼容 `/embeddings` 地址 |
| `dashscope_compatible` | — | 用 OpenAI Client 调用 DashScope 兼容地址 |

`LLM_JSON_MODE=auto` 时，OpenAI 与 DashScope 使用原生 JSON Mode，未知兼容服务只依赖严格 Prompt 和本地 JSON 校验；确认服务支持 `response_format` 后可改为 `native`。结构化长文本默认关闭 DashScope thinking，避免推理 token 占满输出预算而截断 JSON。

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
- Web 会话会在每一幕完成后写入本机 SQLite，默认文件为 `data/scenechat.db`；可通过 `SCENECHAT_DB_PATH` 修改位置。后端重启后可从首页继续，不要求用户安装或操作数据库。
- SQLite 文件包含导演信息、人物秘密和私人记忆，内容未额外加密；不要将该文件提交到仓库或放入公开同步目录。项目已默认忽略数据库及 WAL/SHM 文件。
- 推演页可随时导出一份完整 JSON 档案自行保存；SQLite 恢复与 JSON 导出互不替代。
- 单次推演默认安全上限为 120 轮，可通过 `SCENECHAT_MAX_TURNS` 调整（有效范围 1–1000）；每页请求仍会受批次上限和剩余总轮数限制。
- `/api/story/next-stream` 支持稳定 `request_id` 和 `expected_page`；完成请求可安全重放，断流重试会从已经提交的消息继续，不会重复推进世界状态。
- 新页面请求还可携带 `expected_revision`；状态已经变化时返回 `409 state_revision_conflict`。同一会话生成期间会拒绝并发状态操作，旧客户端不传版本号时仍保持兼容。
- `POST /api/story/session/<id>/interventions/preview` 只生成预检结果，不改变剧情；确认、取消和节奏更新接口都接受 `expected_revision`，并与页面生成共享同一会话操作锁。
- 柔性引导支持“下一步”“持续若干轮”和“持续到取消”。事件注入只能执行安全的公共状态/移动更新；涉及既有事实、角色状态或阶段的强制改写必须由用户在冲突卡上二次确认。
- 干预预检同时读取固定事实和等待执行的干预。对同一状态字段给出互斥结果时，代码层会阻止后提交的普通事件静默覆盖；用户可以先取消旧干预，或明确确认强制改写。
- 仅观众可见的镜头统一作为 `audience_only` 事件进入时间线，不写入任何角色观察或公共状态。公开事件会按可见范围自动形成观察，不接受模型额外注入角色知识。
- 节奏滑块是导演软控制：慢速保留更多反应和关系细节，快速提高关键事件密度并更早寻求自然收束，但不会绕过阶段规则、结束条件、固定设定或角色信息边界。
- `scenechat.evaluation` 提供可重复的导演干预生命周期、时间线可追踪性、私密镜头隔离、patch 边界以及慢/快节奏对照指标，便于部署方在自己的模型配置上进行质量回归。
- 普通浏览器会话 API 只返回公共世界和公开角色卡；用户主动调用导出功能时，下载内容会包含完整导演设定、人物秘密、私人记忆、状态和全部事件历史，请妥善保管。
- 首页的单项删除和“清除全部”会永久删除对应 SQLite 记录；执行前会再次确认。右上角“保存并退出”只退出页面，不删除推演。
- API 默认只接受本机 Vite 前端来源；远程部署时使用 `SCENECHAT_CORS_ORIGINS` 明确填写实际前端来源，不建议配置为 `*`。
- 持久化与导出格式当前为 schema v2；v1 存档在恢复时会自动补齐默认 revision、剧情弧和空干预记录。
- API 错误使用稳定的 `code`、`stage` 和中文 `message`；供应商原始错误与请求 ID 只记录在后端日志中，不直接显示给前端用户。
- 世界观、角色和每轮行动都会调用外部模型，请关注服务商额度和费用。
- 每次点击“预检干预”会额外调用一次文本模型；返回修改但不重新预检不会产生新调用。
