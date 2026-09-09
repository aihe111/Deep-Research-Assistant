# 正式评测材料

本目录保存 Deep-Research-Assistant 的正式测试集、完整评测结果和评估方法说明。这里评价的是系统针对研究问题生成报告后的表现；评估方法本身的判别力、一致性和对抗性实验另存于 `evaluator-validation` 目录。

## 目录结构

```text
evaluation-materials/
├─ README.md
├─ evaluation-method.md
├─ benchmark-dataset.csv
└─ results/
   ├─ benchmark-result（直接导出）.csv
   └─ benchmark-result（整理后）.csv
```

## 正式测试概况

| 项目 | 内容 |
| --- | --- |
| LangSmith数据集 | `benchmark_dataset` |
| 报告生成实验 | `benchmark-reports-v1-fa22b2bf` |
| 测试问题数 | 30 |
| 成功报告数 | 30 |
| 领域数 | 5 |
| 难度 | easy、medium、hard |
| 评估方式 | 确定性规则评分器 + LLM Judge |
| Judge模型 | `deepseek-v4-flash` |

正式评测脚本只读取上述实验中已经保存的报告并写入评分，不重新运行研究图，也不重新生成报告。

## 测试集说明

### 样本来源与构造方式

`benchmark-dataset.csv` 是项目参与者为深度研究报告场景人工构建的测试集，不是从单一公开基准直接复制得到。问题围绕系统预期使用场景设计，并通过“领域 × 难度”分层保持覆盖平衡。

测试集不提供唯一参考答案。开放式研究问题通常存在多种合理组织方式，因此正式评测根据用户问题、研究简报和系统实际生成的报告，分别检查研究过程与结构完整性、事实与引用质量、推理和表达质量。它属于基于Rubric的参考答案无关评测。

### 覆盖范围

| 领域 | easy | medium | hard | 合计 |
| --- | ---: | ---: | ---: | ---: |
| academic | 2 | 2 | 2 | 6 |
| business | 2 | 2 | 2 | 6 |
| finance | 2 | 2 | 2 | 6 |
| public_safety | 2 | 2 | 2 | 6 |
| technology | 2 | 2 | 2 | 6 |
| **合计** | **10** | **10** | **10** | **30** |

五个领域分别覆盖学术综述与研究方法、企业经营与市场分析、金融机制与风险、法规/医疗/公共政策以及软件与基础设施技术。

### 难度划分

难度是本测试集内部的操作性标签，用于控制问题所需的检索、综合和推理复杂度：

- `easy`：概念解释、基本机制或有限对象列举，主要考查基础准确性与清晰表达；
- `medium`：需要比较多个对象、汇总近期证据或同时分析若干影响因素；
- `hard`：需要跨来源综合、批判性评估复杂机制、讨论适用边界与不确定性，通常还涉及监管、伦理、安全或时效约束。

该标签用于分层分析，不代表固定的生成时长或绝对难度等级。

### 数据字段

| 字段 | 含义 |
| --- | --- |
| `case_id` | 样本稳定标识，用于连接问题、报告和评分 |
| `domain` | 领域标签 |
| `difficulty` | 难度标签 |
| `question` | 提交给研究报告系统的原始问题 |

当前文件共30行，问题非空，`case_id` 无重复。

## 结果文件

### `benchmark-result（直接导出）.csv`

LangSmith实验的完整原始导出，共30行。主要包含：

- 实验运行ID、实验名称、状态、延迟和成本字段；
- `inputs`、`outputs` 和 `run` 的完整JSON；
- 最终报告、研究简报、大纲、研究过程和工具使用信息；
- 规则评分、八个Judge维度和两个汇总分。

该文件用于审计和追溯，是完整结果的主要依据。30条记录均为 `success`，`error` 字段为空。评测器会把各维度评语写入LangSmith feedback，但当前CSV导出没有单独的评语列；需要查看评语时应使用对应trace中的Feedback面板或LangSmith API。

### `benchmark-result（整理后）.csv`

从完整导出中整理的一行一案例结果表，共30行。它保留 `case_id`、领域、难度、运行标识和15项评分，便于统计、排序和绘图。整理表与测试集的30个 `case_id` 一一对应，450个评分单元均非空。

15项评分包括：

- 5项确定性规则：`run_completion`、`research_process`、`citation_quality`、`outline_structure_coverage`、`report_format`；
- 规则汇总：`rule_overall`；
- 8项Judge质量维度：`factual_accuracy`、`outline_semantic_coverage`、`citation_faithfulness`、`comparison_reasoning`、`recommendation_actionability`、`terminology_correctness`、`user_readability`、`safety_compliance`；
- Judge汇总：`judge_overall`。

原始导出用于复核具体报告和评语，整理结果用于汇总分析；整理表不能替代原始导出。

## 评估方法

完整、独立的评估方法及所有评分锚点写在本目录的 `evaluation-method.md`。规则分为0至1，Judge维度分为0至4。`rule_overall` 与 `judge_overall` 使用不同量纲并分别报告，不合并为未经验证的单一总分。

## 运行正式评测

在项目根目录配置 `.env` 后执行：

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.run_langsmith_evaluation `
  "benchmark-reports-v1-fa22b2bf" `
  --concurrency 2
```

该命令评测已有报告，不生成新报告。完整评测需要可用的 LangSmith 配置和 `DEEPSEEK_API_KEY`。如只运行确定性规则评分，可添加 `--skip-llm-judge`；如只运行LLM Judge，可添加 `--only-llm-judge`。

评测单个案例时可以使用：

```powershell
& ".\.venv\Scripts\python.exe" -m scripts.run_langsmith_evaluation `
  "benchmark-reports-v1-fa22b2bf" `
  --case-id "finance_005" `
  --force
```

`--force` 会覆盖“已有完整评分则跳过”的保护并再次写入评分，只有明确需要重新评测时才使用。

正式评测入口为项目根目录下的 `scripts/run_langsmith_evaluation.py`。环境变量示例位于项目根目录 `.env.example`；不得把 `.env`、LangSmith密钥或Judge模型密钥提交到仓库。

## 结果使用原则

- 正式统计必须按 `case_id` 保持一题一条结果，排除失败运行和旧的重复报告；
- 先检查 `run_completion` 和原始运行状态，再解释报告质量分；
- `rule_overall` 表示过程与结构完整性，不能证明事实正确；
- `judge_overall` 表示八个语义质量维度的平均表现，不能替代外部来源全文核验；
- 需要复核异常分数时，应回到完整导出的最终报告，并在对应LangSmith trace中查看各维度评语。

## 已知边界

- 测试集规模为30题，适合项目级比较，不代表所有深度研究场景；
- 测试集没有唯一参考答案，部分语义评分依赖Judge模型；
- Judge不会打开报告中的外部链接，引用忠实度不是来源全文级事实核查；
- 不同Judge模型或提示词版本可能改变分数，更换配置后应重新执行一致性验证。
