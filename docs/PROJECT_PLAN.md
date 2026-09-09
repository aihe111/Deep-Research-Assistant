# Deep-Research-Assistant 架构说明

## 产品定位

Deep-Research-Assistant 是任意领域的通用深度研究 Agent。领域能力来自模型规划与可动态扩展的工具，
而不是写死的论文检索步骤。系统输出带来源的 Markdown 报告，并保留 Thread、Run、工具调用和
研究产物之间的追溯关系。

## 分层

1. FastAPI/Web：会话、交互、报告阅读与 LangGraph Thread/Run 调用。
2. Main Graph：澄清、研究大纲、Manager 子图、最终报告。
3. Manager：模型反思研究覆盖，批量委派独立研究单元并决定何时完成。
4. Researcher：按研究单元循环执行策略反思、工具调用、质量反思和证据压缩。
5. Tool Layer：OpenAlex、Tavily 与动态 MCP；保留 provider-native 来源结构。
6. Persistence/Observability：LangGraph checkpoint、应用数据库产物、LangSmith traces/evals。

## 关键边界

- 并行上限、Manager 循环和 Researcher 循环均由配置显式限制，避免失控调用。
- 单个工具失败会作为 ToolMessage 返回给模型，不会丢弃其他并行研究单元。
- 模型负责语义质量判断，但引用只能来自工具返回的证据；最终提示词禁止补造来源。
- Web 的 Thread ID 同时作为应用会话 ID 与报告 artifact key。
- MCP 工具无公开 URL 时保留 server/tool 和原生 ID，不伪造链接。

## 后续里程碑

1. 为重点领域增加 LangSmith gold dataset 和来源权威性 LLM judge。
2. 为 MCP Server 增加 allowlist、OAuth 与敏感字段脱敏策略。
3. 根据线上 trace 优化并发、模型分层和上下文压缩成本。
4. 部署持久化 Agent Server，并验证 Run 重试、取消、超时和水平扩展。
