from deep_research_assistant.intent_normalizer import normalize_explicit_constraints
from deep_research_assistant.models import ReportDepth, ReportLanguage, ResearchIntent


def test_normalizer_preserves_explicit_range_audience_and_language() -> None:
    model_intent = ResearchIntent(
        topic="RAG 评测",
        goal="梳理方法",
        audience="AI 与计算机领域学习者",
        end_year=2026,
    )

    result = normalize_explicit_constraints(
        "调研 2023-2026 年 RAG 评测，面向有基础的大模型开发者，"
        "重点关注检索质量，生成3000字中文标准深度报告",
        model_intent,
    )

    assert result.start_year == 2023
    assert result.end_year == 2026
    assert result.audience == "有基础的大模型开发者"
    assert result.report_language is ReportLanguage.CHINESE
    assert result.depth is ReportDepth.STANDARD
    assert result.target_word_count == 3000
    assert result.focus_areas == ["检索质量"]
    assert not result.clarification_questions


def test_normalizer_preserves_explicit_source_count() -> None:
    model_intent = ResearchIntent(topic="Agent 记忆", goal="形成综述")

    result = normalize_explicit_constraints("筛选 12 篇核心文献并形成英文深度报告", model_intent)

    assert result.target_source_count == 12
    assert result.report_language is ReportLanguage.ENGLISH
    assert result.depth is ReportDepth.DEEP


def test_normalizer_requests_missing_required_parameters() -> None:
    model_intent = ResearchIntent(topic="RAG", goal="形成综述")

    result = normalize_explicit_constraints("调研 RAG", model_intent)

    assert len(result.clarification_questions) == 4
    assert any("多少字" in question for question in result.clarification_questions)
    assert any("目标读者" in question for question in result.clarification_questions)


def test_normalizer_does_not_block_on_optional_source_type() -> None:
    model_intent = ResearchIntent(
        topic="RAG 评测",
        goal="形成综述",
        clarification_questions=["希望包含哪些来源类型？"],
    )

    result = normalize_explicit_constraints(
        "调研 2023-2026 年 RAG 评测，面向开发者，重点关注忠实性，生成3000字报告",
        model_intent,
    )

    assert not result.clarification_questions
