from deep_research_assistant.literature_ranker import (
    deduplicate_works,
    rank_works,
    section_coverage,
    select_with_section_coverage,
)
from deep_research_assistant.models import LiteratureQuery, ScholarlyWork


def _work(**overrides: object) -> ScholarlyWork:
    values: dict[str, object] = {
        "openalex_id": "https://openalex.org/W1",
        "title": "Evaluating Retrieval Augmented Generation Systems",
        "abstract": "Evaluation of retrieval quality and answer faithfulness in RAG systems.",
        "publication_year": 2024,
        "cited_by_count": 25,
        "matched_query_ids": ["Q1"],
    }
    values.update(overrides)
    return ScholarlyWork.model_validate(values)


def test_deduplicate_merges_query_provenance() -> None:
    first = _work(matched_query_ids=["Q1"])
    duplicate = _work(
        openalex_id="https://openalex.org/W2",
        doi="https://doi.org/10.1/rag",
        title="Evaluating Retrieval-Augmented Generation Systems",
        matched_query_ids=["Q2"],
    )

    result = deduplicate_works([first, duplicate])

    assert len(result) == 1
    assert result[0].doi == "https://doi.org/10.1/rag"
    assert result[0].matched_query_ids == ["Q1", "Q2"]


def test_ranker_prefers_stronger_lexical_match() -> None:
    query = LiteratureQuery(
        query_id="Q1",
        section_id="S1",
        research_question="如何评价 RAG？",
        query="retrieval augmented generation evaluation",
        rationale="覆盖核心主题",
    )
    relevant = _work()
    weak = _work(
        openalex_id="https://openalex.org/W3",
        title="General Language Model Survey",
        abstract="A broad survey of language models.",
        cited_by_count=200,
    )

    ranked = rank_works([weak, relevant], [query], start_year=2023, end_year=2026)

    assert ranked[0].openalex_id == relevant.openalex_id
    assert ranked[0].relevance_score > ranked[1].relevance_score


def test_selector_reserves_a_work_for_each_searchable_section() -> None:
    queries = [
        LiteratureQuery(
            query_id=f"Q{index}",
            section_id=f"S{index}",
            research_question="需要什么证据？",
            query=f"research topic {index}",
            rationale="覆盖章节",
        )
        for index in range(1, 4)
    ]
    works = [
        _work(
            openalex_id=f"https://openalex.org/W{index}",
            title=f"Study {index}",
            matched_query_ids=[f"Q{index}"],
            relevance_score=1 - index / 10,
        )
        for index in range(1, 4)
    ]

    selected = select_with_section_coverage(works, queries, limit=3)

    assert section_coverage(selected, queries) == (3, 3)
