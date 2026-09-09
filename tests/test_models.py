import pytest
from pydantic import ValidationError

from deep_research_assistant.models import ReportLanguage, ResearchIntent, SourceType


def test_research_intent_accepts_valid_year_range() -> None:
    intent = ResearchIntent(topic="RAG 评测", goal="梳理主要方法", start_year=2023, end_year=2026)

    assert intent.target_source_count == 8
    assert intent.report_language is ReportLanguage.CHINESE
    assert intent.source_types == [SourceType.PAPER]


def test_research_intent_rejects_reversed_year_range() -> None:
    with pytest.raises(ValidationError, match="start_year"):
        ResearchIntent(topic="Agent 记忆", goal="梳理技术路线", start_year=2026, end_year=2020)
