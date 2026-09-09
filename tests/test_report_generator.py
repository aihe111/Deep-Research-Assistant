import pytest

from deep_research_assistant.models import (
    LiteratureSearchPlan,
    LiteratureSearchResult,
    OutlineSection,
    ResearchIntent,
    ResearchOutline,
    ScholarlyWork,
)
from deep_research_assistant.report_generator import (
    ReportCitationError,
    append_deterministic_references,
    validate_citations,
)


def _search_result() -> LiteratureSearchResult:
    outline = ResearchOutline(
        title="RAG 调研",
        thesis="如何评价 RAG",
        sections=[
            OutlineSection(
                section_id=f"S{index}",
                title=f"章节 {index}",
                objective="整理证据",
                research_questions=["有哪些方法？"],
            )
            for index in range(1, 4)
        ],
    )
    return LiteratureSearchResult(
        intent=ResearchIntent(topic="RAG 评测", goal="梳理评测方法"),
        outline=outline,
        search_plan=LiteratureSearchPlan(
            queries=[
                {
                    "query_id": f"Q{index}",
                    "section_id": f"S{index}",
                    "research_question": "有哪些方法？",
                    "query": "RAG evaluation",
                    "rationale": "覆盖主题",
                }
                for index in range(1, 4)
            ]
        ),
        retrieved_count=1,
        deduplicated_count=1,
        works=[
            ScholarlyWork(
                openalex_id="https://openalex.org/W1",
                doi="https://doi.org/10.1/example",
                title="RAG Evaluation",
                authors=["Ada Researcher"],
                publication_year=2024,
                source_name="Example Journal",
                abstract="A framework for RAG evaluation.",
            )
        ],
    )


def test_append_references_uses_retrieved_metadata() -> None:
    report = append_deterministic_references(
        "# 报告\n\n已有研究提出评测框架。[REF001]", _search_result()
    )

    assert "## 参考文献" in report
    assert "Ada Researcher. (2024). RAG Evaluation." in report
    assert "https://doi.org/10.1/example" in report


def test_validate_citations_rejects_unknown_reference() -> None:
    with pytest.raises(ReportCitationError, match="未知引用"):
        validate_citations("错误引用。[REF999]", {"REF001"})


def test_validate_citations_requires_at_least_one_reference() -> None:
    with pytest.raises(ReportCitationError, match="没有引用"):
        validate_citations("没有引用的报告。", {"REF001"})


def test_reference_list_is_sorted_by_reference_number() -> None:
    search_result = _search_result()
    second_work = search_result.works[0].model_copy(
        update={
            "openalex_id": "https://openalex.org/W2",
            "doi": "https://doi.org/10.1/second",
            "title": "Second Study",
        }
    )
    search_result = search_result.model_copy(
        update={"works": [search_result.works[0], second_work]}
    )

    report = append_deterministic_references(
        "# 报告\n\n先引用第二篇。[REF002] 再引用第一篇。[REF001]",
        search_result,
    )
    references = report.split("## 参考文献", maxsplit=1)[1]

    assert references.index("[REF001]") < references.index("[REF002]")
