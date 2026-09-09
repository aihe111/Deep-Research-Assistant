# 评估方法有效性验证

本目录保存 Deep-Research-Assistant 评估方法的有效性验证材料。验证对象不是研究报告生成能力本身，而是用于评价报告的“规则评分器 + LLM Judge”能否区分质量差异、对同一输出保持稳定，并识别典型投机性扰动。

完整实验方法、统计结果、结论与限制见 [`evaluator-validation-report.md`](./evaluator-validation-report.md)。

## 目录内容

```text
evaluator-validation/
├─ README.md
├─ evaluator-validation-report.md
├─ Discernment-Check/
│  ├─ benchmark-discernment-samples.csv
│  ├─ benchmark-discernment-result.csv
│  └─ benchmark-discernment-result.png
├─ Consistency-Check/
│  ├─ benchmark-consistency-samples.csv
│  └─ benchmark-consistency-result.csv
└─ Adversarial-Check/
   ├─ benchmark-adversarial-samples.csv
   └─ benchmark-adversarial-result.csv
```

`Discernment-Check` 对应判别力验证（discrimination validation）。该目录名为现有文件结构中的名称，报告中统一称为“判别力验证”。

## 验证内容

| 验证 | 样本与运行 | 目的 | 当前结果 |
| --- | ---: | --- | --- |
| 判别力验证 | 3个主题 × 好/中/差3档，共9份报告 | 检查评估方法能否正确区分并排序明显不同的质量档位 | 3组三档顺序全部正确；9/9成对排序正确；Spearman ρ=0.982 |
| 一致性验证 | 30份冻结报告 × 3轮，共90条结果 | 检查同一模型、提示词和输入重复评分时的波动 | 八维完全一致率93.33%；相差不超过1分比例100% |
| 对抗性验证 | 3份定向扰动报告 | 检查增加篇幅、堆砌术语和伪造引用是否会被错误奖励 | 3/3达到预设降分条件；属于初步验证 |

## 评估方法

评估分为两层，二者分别报告，不合并为一个未经验证的总分。

### 确定性规则评分

规则评分取值为0至1，包含：

- `run_completion`：运行状态、报告非空和错误信息；
- `research_process`：研究单元、工具调用、检索工具和来源数量；
- `citation_quality`：编号引用结构、正文与来源对应关系和可追溯地址；
- `outline_structure_coverage`：报告标题与大纲章节的结构匹配；
- `report_format`：Markdown标题、正文层级、主要来源章节和失败占位语；
- `rule_overall`：上述规则的加权汇总。

在判别力样本中没有研究过程字段，因此 `research_process` 不参与该实验的规则总分，并对其余可用规则权重重新归一化。

### LLM Judge质量评分

LLM Judge按0至4整数评分八个维度：事实准确性、大纲语义覆盖、引用忠实度、比较推理、建议可执行性、术语正确性、用户可理解性和安全合规性。`judge_overall` 是八个维度的等权算术平均，满分4分。

本次验证使用：

- Judge模型：`deepseek-v4-flash`；
- 温度：0；
- Thinking：关闭；
- 最大输出：1500 tokens；
- 评估器版本：`70f36d7b8c516da1`（一致性与对抗性结果中已记录）。

## 数据来源

一致性验证使用正式测试集生成的30份冻结报告。正式测试集位于相邻目录 [`../evaluation-materials/benchmark-dataset.csv`](../evaluation-materials/benchmark-dataset.csv)，包含5个领域，每个领域6题；easy、medium、hard各10题。冻结报告及正式评分保存在 `evaluation-materials/results/`。

判别力和对抗性样本为人工定向构造的验证样本，不计入正式测试集成绩：

- 判别力样本分别构造好、中、差三个质量档位；
- 对抗样本从正式报告派生，仅改变报告内容，不重新生成基线报告。

## 复现实验

以下命令在项目根目录执行。需要在本地环境中配置 LangSmith 和 Judge 模型所需的环境变量；不得提交 `.env` 或任何密钥。

```powershell
# 判别力验证
& ".\.venv\Scripts\python.exe" -m scripts.run_evaluator_calibration `
  "deep-research-evaluator-calibration-v1" `
  --prefix "evaluator-calibration-full-v1" `
  --repetitions 1

# 一致性验证：只读取冻结报告，不重新生成报告
& ".\.venv\Scripts\python.exe" -m scripts.run_consistency_evaluation `
  "benchmark_consistency_v1" `
  --prefix "benchmark-consistency-trial" `
  --rounds 3 `
  --expected-count 30

# 对抗性验证：只读取扰动后的冻结报告
& ".\.venv\Scripts\python.exe" -m scripts.run_adversarial_evaluation `
  "benchmark_adversarial_v1" `
  --expected-count 3
```

相关脚本位于项目根目录：

- [`../scripts/run_evaluator_calibration.py`](../scripts/run_evaluator_calibration.py)
- [`../scripts/run_consistency_evaluation.py`](../scripts/run_consistency_evaluation.py)
- [`../scripts/run_adversarial_evaluation.py`](../scripts/run_adversarial_evaluation.py)
- [`../scripts/run_langsmith_evaluation.py`](../scripts/run_langsmith_evaluation.py)

## 结果解释

- 判别力和一致性满足当前验证目标，支持将该评估方法用于本项目的30份正式报告。
- 对抗性验证仅覆盖3种扰动且每种1例，只能作为初步证据。
- 本验证没有完成独立双人标注及人工—Judge一致性检验，不应表述为对评估方法普遍有效性的充分证明。
- 引用忠实度基于报告内引用位置、来源标题和地址进行判断；Judge没有读取链接正文，因此不能替代逐条来源全文核验。

## 提交注意事项

- 保留CSV原始数据和本报告，不要只提交截图。
- `benchmark-discernment-result.png` 仅作为界面展示，不作为统计数据来源。
- 不提交 `.env`、API密钥、缓存目录、临时导出或包含错误记录的重复实验。

