# Deep-Research-Assistant

> 个人 / 犀牛鸟开源实战活动作品，非腾讯官方发布。

Deep-Research-Assistant 是一个基于 Hy3、LangGraph 和多来源工具的通用深度研究 Agent。它不限定于
AI 论文调研：科技、商业、政策、产品、医学等领域都由同一套 Manager / Researcher 架构处理。
FastAPI 提供产品页面，耗时任务以 Thread/Run 交给 LangGraph Server 后台执行。

系统提供中文 Web、会话记忆、报告产物和 Hy3 OpenAI-compatible 接口，并通过多智能体协作完成
从需求澄清、研究规划、证据检索到报告交付的完整流程。

## 核心能力

- 在开始检索前澄清关键歧义，并让用户确认或修改研究大纲。
- 由 Manager 动态拆分任务，支持多个 Researcher 并行检索和证据反思。
- 同时装载 Tavily、OpenAlex 全文工具和可配置的 MCP 工具。
- 对网页、论文和工具结果去重、压缩，最终生成带可追溯来源的中文 Markdown 报告。
- 通过 LangGraph Thread/Run 后台执行，通过 SSE 在 Web 页面展示研究过程和报告增量。
- 使用 SQLite 保存会话、消息、事件和报告，并可接入 LangSmith trace 与离线评测。

## 研究流程

```mermaid
flowchart TD
    start([START]) --> clarify["澄清研究意图<br/>clarify_with_user"]
    clarify --> outline["生成研究大纲<br/>write_research_outline"]
    outline --> confirm{"确认研究大纲<br/>confirm_outline_with_user"}
    confirm -- 修改 --> outline
    confirm -- 确认 --> manager_start

    subgraph research_manager["research_manager 子图"]
        direction TB
        manager_start([START]) --> manager[Manager]
        manager -- 调用调度工具 --> tools[Tools]
        tools -- 返回研究结果 --> manager
        manager -- 完成复核 --> manager_end([END])

        tools -- 并行派发 --> researcher_1["Researcher 1<br/>检索 · 反思 · 压缩"]
        tools -- 并行派发 --> researcher_2["Researcher 2<br/>检索 · 反思 · 压缩"]
        tools -- 并行派发 --> researcher_n["Researcher N<br/>检索 · 反思 · 压缩"]
        researcher_1 --> tools
        researcher_2 --> tools
        researcher_n --> tools
    end

    manager_end --> report["生成完整报告<br/>report_generation"]
    report --> finish([END])
```

LangGraph 运行时生成的核心工作流如下：

![Deep-Research-Assistant 核心工作流](docs/images/core-workflow.png)

- `clarify_with_user` 只在存在会实质改变研究方向的歧义时询问一次。
- `write_research_outline` 把会话转换成可独立委派的研究简报和结构化大纲。
- `confirm_outline_with_user` 暂停任务等待确认；修改意见返回简报节点重新生成，默认最多 3 次。
- Manager 可在一次工具调用中提交多个 `ConductResearch`，Researcher 通过
  `asyncio.gather` 并行执行，数量由 `MAX_CONCURRENT_RESEARCH_UNITS` 控制；比较类任务默认倾向
  为每个比较对象分别启动一个 Researcher，必要时再增加跨对象的标准或背景研究单元；非比较类
  任务仍优先只启动一个 Researcher。
- 每个 Researcher 会先后使用 `think_tool` 反思检索策略与证据质量。相关性、权威性、时效、
  冲突和缺口由模型语义判断，不再使用固定分数或篇数质量门。
- 简单问题使用 2～3 次搜索，复杂问题最多 5 次；取得 3 个以上直接相关来源并足以回答，或连续
  两次结果高度相似时停止。搜索与反思交替执行，因此 Researcher 默认最多运行 10 个 ReAct 回合。
- 最终报告综合各研究单元的完整研究发现，根据正文实际需要引用来源，不设置固定来源数量，也不
  简单截取排名前几项。来源按首次引用顺序生成 `[1]`、`[2]` 等全局连续编号；生成后自动检查
  缺号、漏列和截断，来源列表按编号升序且每条独占一行。不完整的报告会关闭思考模式自动重写一次。

## 研究工具与来源规范

工具在 Researcher 运行时装载，模型按问题选择工具：

| 工具 | 用途 | 来源与引用规范 |
| --- | --- | --- |
| OpenAlex | 论文、综述、学术证据 | 先检索元数据与摘要，再按 Work ID 从官方 Content API 读取 GROBID XML/PDF 正文；明确标记摘要级或全文级证据 |
| Tavily | 通用和时效性 Web 检索 | 并发查询后按规范化 URL 去重，保留全部非重复网页；以标题 + URL 引用 |
| MCP | 内部数据、第一方系统、专业服务 | 动态保留 server/tool、原生记录 ID 与 URL；没有公开 URL 时明确标为 MCP/私有来源 |

OpenAlex 已从固定流水线步骤改成 LangChain Tool。设置 `MCP_SERVERS` 后，
`MultiServerMCPClient` 会在运行时从多个 MCP Server 发现工具，无需修改图代码。

OpenAlex 分成两个工具：`openalex_search` 负责发现论文和摘要，
`openalex_fetch_fulltext` 只对 Researcher 选中的 Work ID 读取正文。全文来自 OpenAlex 官方
Content API，优先使用结构化 GROBID XML，失败后回退 PDF + PyMuPDF。该接口要求 API Key，
每次内容下载会消耗 OpenAlex Content API 额度，因此默认每次最多处理三篇，且不会绕过付费墙。

Tavily 工具支持批量查询并启用 `include_raw_content=True`。每个查询默认请求
最多 20 条结果；候选结果仍按相关性排序，但排序只决定处理顺序，不再截取 Top-K。跨查询按规范化
URL 去重后，全部非重复网页都会进入结果。为控制 Token，默认只对排名靠前的 5 个页面额外执行
全文总结，其余页面仍完整保留标题、原始 URL 和搜索片段。进程级 LRU 缓存让多个 Researcher
再次遇到同一网页时复用摘要。默认每页最多向总结模型输入 20000 字符；总结失败或全局预算耗尽时
同样保留搜索片段，不会丢弃网页，也不会把大段原文直接塞回 Researcher 上下文。

默认不设置单次研究的 Hy3 全局硬调用次数上限（`MAX_HY3_CALLS_PER_RESEARCH=0`），但仍记录总调用
数和分阶段调用数。Manager 轮数、Researcher ReAct 轮数、单个 Researcher 搜索次数、并发数以及
模型重试次数仍分别受限，避免无限循环。部署方如需成本硬保护，可把该配置改为正整数；此时系统
会恢复全局预算、终端调用保护和每个研究单元的压缩调用预留。Researcher 的单个工具结果默认压缩到
20000 字符、每轮模型上下文限制为
60000 字符；进入压缩模型的证据最多 50000 字符。Manager 默认最多输出 6000 tokens；Manager
或 Researcher 同时缺少正文和工具调用时，会额外调用一次关闭思考的模型重试，空响应不再被当作
正常完成。证据压缩关闭思考模式，默认最多输出 12000 tokens，生成的完整研究发现目标约
7000 字符、硬上限为 10000 字符；模型异常返回空内容时自动保留确定性证据摘要。最终报告默认
最多输出 16000 tokens，空响应时关闭思考重试一次。最终状态会记录实际调用数、分阶段调用数；
仅当部署方显式配置正整数全局上限时，才可能触发预算耗尽状态，便于在
LangSmith 和产品 Web 中审计。

## 环境要求

- Python 3.11（推荐；`langgraph.json` 也固定为 3.11）。
- `uv`/`uvx`，用于启动本地 LangGraph 开发服务器。
- 腾讯云 TokenHub 的 Hy3 API Key。
- Tavily、OpenAlex、LangSmith、DeepSeek 和 MCP 均为按功能选配。

## 安装

Windows PowerShell：

```powershell
Set-Location -LiteralPath "Deep-Research-Assistant"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip uv
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Linux / macOS：

```bash
cd Deep-Research-Assistant
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv
python -m pip install -e ".[dev]"
cp .env.example .env
```

随后编辑本地 `.env`。最小必填配置只有：

```dotenv
HY3_API_KEY=replace_with_your_tokenhub_key
```

建议至少再配置一个网页搜索入口；Tavily Hub 与官方 Tavily 二选一即可，Hub Key 优先：

```dotenv
TAVILY_HUB_API_KEY=
TAVILY_API_KEY=
OPENALEX_API_KEY=
```

Tavily Hub 默认通过 `https://tavily.sharyuke.com/api/proxy` 调用。它是第三方中转服务，请勿通过它
检索敏感或私密内容。OpenAlex 元数据检索可匿名运行；读取 OpenAlex Content API 全文需要 Key，
并会消耗相应接口额度。

MCP 多服务器配置示例：

```dotenv
MCP_SERVERS={"docs":{"transport":"streamable_http","url":"http://localhost:8001/mcp"},"local":{"transport":"stdio","command":"python","args":["my_server.py"]}}
MCP_PROMPT=优先使用 docs 获取第一方产品资料，使用 local 查询内部数据。
```

完整配置项及安全占位值见 [.env.example](.env.example)。真实 `.env` 已被 `.gitignore` 排除，
不得提交到 GitHub。

## 启动 Web 应用

需要同时运行 LangGraph Server 和产品 Web。以下两个命令应分别放在两个终端中，并保持终端开启。

终端 1（Windows PowerShell）启动 LangGraph Server：

```powershell
Set-Location -LiteralPath "Deep-Research-Assistant"
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
& ".\.venv\Scripts\uvx.exe" --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

Linux / macOS 对应命令：

```bash
cd Deep-Research-Assistant
source .venv/bin/activate
PYTHONUTF8=1 .venv/bin/uvx --refresh --from 'langgraph-cli[inmem]' --with-editable . --python 3.11 langgraph dev --allow-blocking
```

看到 `Server started` 后，再在终端 2 启动产品 Web：

```powershell
Set-Location -LiteralPath "Deep-Research-Assistant"
.\.venv\Scripts\Activate.ps1
deep-research-assistant doctor
deep-research-assistant web
```

浏览器访问 <http://127.0.0.1:8000>。可以分别用以下地址确认服务状态：

- LangGraph：<http://127.0.0.1:2024/ok>
- 产品 Web：<http://127.0.0.1:8000/api/health>

提交问题时，Web 创建 Run 并立即返回 HTTP 202；LangGraph Server 在后台排队执行。页面通过 SSE
实时接收研究过程和报告增量，断线后按事件 ID 续传，并在 SSE 不可用时退回状态轮询。澄清和大纲
确认使用 LangGraph `interrupt` 暂停，用户回答后由新的 Resume Run 从 checkpoint 接续。

开发命令使用的是 `langgraph-cli[inmem]`：关闭 LangGraph Server 后，Thread、Run 和 checkpoint
会丢失。会话、消息、过程事件和 Markdown 报告独立保存在 `DATABASE_URL`，默认是本地 SQLite。
生产环境应使用带持久化后端的 LangGraph 部署；应用数据库可改成 PostgreSQL。

### 可选：本地 PostgreSQL

SQLite 足以完成本地演示。需要验证 PostgreSQL 时，可以使用仅绑定本机端口的开发容器：

```powershell
docker compose up -d postgres
```

然后在 `.env` 中修改：

```dotenv
DATABASE_URL=postgresql+psycopg://deep_research_assistant:deep_research_assistant_local_only@localhost:5432/deep_research_assistant
```

`docker-compose.yml` 中的默认账号仅供本地开发；对外部署前必须设置新的
`POSTGRES_USER`、`POSTGRES_PASSWORD` 和网络访问策略。

## 项目结构

```text
Deep-Research-Assistant/
├─ src/deep_research_assistant/   # Agent、LangGraph、工具、存储与 Web 源码
│  └─ web_static/                 # 原生 HTML/CSS/JavaScript 前端
├─ tests/                         # 单元测试与工作流测试
├─ scripts/                       # 冒烟测试、LangSmith 评测与数据集脚本
├─ docs/                          # 架构、方案与评测文档
├─ .env.example                   # 可公开的完整环境配置样例
├─ docker-compose.yml             # 可选的本地 PostgreSQL
├─ langgraph.json                 # LangGraph Server 图入口
└─ pyproject.toml                 # Python 包、依赖和命令入口
```

运行时产生的 `.env`、SQLite 数据库、报告输出、缓存和临时文件均被 `.gitignore` 排除。

## 常见启动问题

- **`UnicodeDecodeError: 'gbk' codec can't decode ...`**：在启动 LangGraph 的同一个 PowerShell
  中设置 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`，再执行 `uvx` 命令。
- **无法识别 `uvx`**：激活虚拟环境后运行 `python -m pip install uv`，重新打开终端再试。
- **`127.0.0.1:2024` 拒绝连接**：LangGraph Server 尚未启动或启动终端已经关闭。
- **Hy3 返回 HTTP 402**：TokenHub 免费额度已经耗尽，需要更换有效额度或在控制台启用计费。
- **重启后旧会话无法继续运行**：开发服务器使用内存 Thread；SQLite 中的报告仍在，但旧 Thread
  不会跨 LangGraph Server 重启保存。

## LangSmith 可观测与评测

开启节点、工具与子研究单元的 LangSmith trace：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=deep-research-assistant
DEEPSEEK_API_KEY=your_deepseek_key
EVALUATION_JUDGE_MODEL=deepseek-v4-flash
```

先完成一次报告生成实验，再把该 LangSmith experiment 的名称或 UUID 传给评测脚本：

```powershell
python scripts/run_langsmith_evaluation.py your-existing-experiment --concurrency 2
```

评测入口只读取现有 experiment 中已经保存的输入、研究简报和最终报告，不会重新创建 Thread、
调用研究图或生成报告。它返回运行完成度、研究过程、编号引用完整性、大纲结构覆盖、报告格式和
加权规则总分六项确定性指标。每篇完成的报告只调用
一次 DeepSeek 评审模型，同时写入事实准确性、大纲语义覆盖度、引用忠实度、比较推理质量、
建议可执行性、专业术语正确性、用户可理解性、安全合规性和八维平均分。评审读取完整的用户问题、
研究简报与最终报告；DeepSeek 思考模式会被显式关闭，以兼容 JSON 结构化输出并减少费用。

只运行不产生模型费用的规则评分器：

```powershell
python scripts/run_langsmith_evaluation.py your-existing-experiment --skip-llm-judge
```

### 评测器校准数据集

若要用人工准备的好、中、差报告验证评分器本身，请单独建立 LangSmith 数据集，例如
`deep-research-evaluator-calibration-v1`。每条样本的 Inputs 必须包含：

- `question`：原始问题；
- `research_brief` 或 `outline`：研究简报/大纲，可为 JSON 对象或 JSON 字符串；
- `final_report` 或 `report`：人工准备的完整报告。

校准脚本把预制报告原样作为实验输出，同时运行规则评分器和八维 LLM Judge，不会调用研究图：

```powershell
.\.venv\Scripts\python.exe -m scripts.run_evaluator_calibration `
  "deep-research-evaluator-calibration-v1" `
  --prefix "evaluator-calibration-v1" `
  --concurrency 1
```

一致性验证可让每个样本重复评测三次：

```powershell
.\.venv\Scripts\python.exe -m scripts.run_evaluator_calibration `
  "deep-research-evaluator-calibration-v1" `
  --prefix "evaluator-calibration-repeat-v1" `
  --concurrency 1 `
  --repetitions 3
```

这会产生“样本数 × 重复次数”次 LLM Judge 调用。若只验证确定性规则，可增加
`--skip-llm-judge`。只有问题、大纲和报告时，脚本不会臆造研究过程数据，
`research_process` 将不评分并从校准规则总分权重中排除。若需要验证该指标，可在 Inputs
中额外提供 `research_unit_count`、`source_count` 和 `tools_used`。

## 兼容功能

- 报告完成后可围绕报告继续追问；追问仍作为独立后台 Run，不重复执行检索。
- 原来的 `plan`、`search`、`report` CLI 保留，供旧的 OpenAlex 文献流水线分阶段调试；Web 和
  LangGraph Server 的默认主图已经切换到通用 Deep Research。
- 会话、消息、滚动摘要和 Markdown 报告继续使用原数据库结构，升级不需要迁移已有数据。

## 验证

```powershell
python -m ruff check src tests scripts
python -m pytest -q
python -c "from deep_research_assistant.server_graph import graph; print(graph.get_graph(xray=True).draw_mermaid())"
```
