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
8. 阶段、位置、资源、技能、关系、目标、投票、淘汰、定向认知和组合结束条件随推演更新。自然结束、连续失败阻断与 `MAX_TURNS` 安全上限彼此区分。

启动实验前先探测生成模型。生成结构化设定后，只有背景长度达到 RAG 阈值时才探测向量模型并建立索引；短场景不会因为 Embedding 不可用而无法运行。

DashScope 文本向量接口每批最多接受 20 条内容。项目在 Embedding 客户端和索引插入层都将批次固定限制为 20，较大的角色/世界文档会自动拆成多批处理。

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
│   ├── errors.py                  # 稳定错误码与对外错误信息
│   ├── generation.py              # 约束账本、世界、角色的分阶段生成
│   ├── knowledge.py               # 实验级隔离索引与角色过滤检索
│   ├── models.py                  # AgentState / Message / SimulationState
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
LLM_API_KEY=your_api_key
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
MODEL_PROVIDER=dashscope

EMBEDDING_API_KEY=your_api_key
EMBEDDING_MODEL=text-embedding-v2
EMBEDDING_PROVIDER=dashscope

LLM_REQUEST_TIMEOUT_SECONDS=180
LLM_MAX_RETRIES=1
LLM_ENABLE_THINKING=false
SIMULATION_PARSE_RETRIES=1
```

当前推演和 Embedding 提供商实现为 DashScope；世界观和角色生成通过其 OpenAI 兼容接口调用。
结构化长文本生成默认关闭 thinking，避免推理 token 占满输出预算而截断 JSON；只有确认所用模型和额度适合时才应显式开启。

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
- Web 会话当前保存在进程内，后端重启后失效，定位仍是单用户本地原型。
- 每页请求会同时受到批次上限和剩余总轮数限制，不会越过 `MAX_TURNS`。
- `/api/story/next-stream` 支持稳定 `request_id` 和 `expected_page`；完成请求可安全重放，断流重试会从已经提交的消息继续，不会重复推进世界状态。
- 浏览器 API 只返回公共世界和公开角色卡；完整导演设定和私密角色档案只保留在后端会话及实验归档。
- API 错误使用稳定的 `code`、`stage` 和中文 `message`；供应商原始错误与请求 ID 只记录在后端日志中，不直接显示给前端用户。
- 世界观、角色和每轮行动都会调用外部模型，请关注服务商额度和费用。
