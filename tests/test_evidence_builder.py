from typing import Any

from pydantic import BaseModel

from deep_research_assistant.evidence_builder import EvidenceBuilder
from deep_research_assistant.models import (
    EvidenceCard,
    EvidenceCollection,
    LiteratureSearchPlan,
    LiteratureSearchResult,
    OutlineSection,
    ResearchIntent,
    ResearchOutline,
    ScholarlyWork,
)


def _search_result() -> LiteratureSearchResult:
    return LiteratureSearchResult(
        intent=ResearchIntent(topic="RAG 评测", goal="梳理评测方法"),
        outline=ResearchOutline(
            title="RAG 评测调研",
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
        ),
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
                title="RAG Evaluation",
                abstract="A framework for evaluating retrieval and generation.",
            )
        ],
    )


class FakeEvidenceClient:
    def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        **_: Any,
    ) -> BaseModel:
        assert response_model is EvidenceCollection
        return EvidenceCollection(
            cards=[
                EvidenceCard(
                    citation_id="REF001",
                    openalex_id="model-copied-id",
                    supported_section_ids=["S1"],
                    relevance_summary="与 RAG 评测直接相关",
                    key_findings=["提出同时评价检索和生成的框架"],
                    limitations=["摘要未提供完整实验设置"],
                )
            ]
        )


def test_evidence_builder_restores_deterministic_source_mapping() -> None:
    builder = EvidenceBuilder(FakeEvidenceClient())  # type: ignore[arg-type]

    result = builder.build(_search_result())

    assert result.cards[0].citation_id == "REF001"
    assert result.cards[0].openalex_id == "https://openalex.org/W1"
