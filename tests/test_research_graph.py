from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from deep_research_assistant.artifact_store import ArtifactStore
from deep_research_assistant.models import (
    EvidenceCard,
    EvidenceCollection,
    LiteratureSearchPlan,
    ResearchIntent,
    ResearchOutline,
    ScholarlyWork,
)
from deep_research_assistant.research_graph import build_research_graph


class FakeHy3Client:
    def __init__(self) -> None:
        self.outline_calls = 0

    def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        **_: Any,
    ) -> BaseModel:
        if response_model is ResearchIntent:
            return ResearchIntent(
                topic="retrieval augmented generation evaluation",
                goal="compare evaluation methods",
                audience="RAG developers",
                focus_areas=["retrieval quality", "faithfulness"],
                start_year=2023,
                end_year=2026,
                target_word_count=1000,
                target_source_count=3,
            )
        if response_model is ResearchOutline:
            self.outline_calls += 1
            return ResearchOutline.model_validate(
                {
                    "title": f"RAG Evaluation Outline {self.outline_calls}",
                    "thesis": "How should RAG systems be evaluated?",
                    "sections": [
                        {
                            "section_id": f"S{index}",
                            "title": f"Dimension {index}",
                            "objective": "Collect comparative evidence",
                            "research_questions": [f"What evidence covers dimension {index}?"],
                        }
                        for index in range(1, 4)
                    ],
                }
            )
        if response_model is LiteratureSearchPlan:
            return LiteratureSearchPlan.model_validate(
                {
                    "queries": [
                        {
                            "query_id": f"Q{index}",
                            "section_id": f"S{index}",
                            "research_question": f"What evidence covers dimension {index}?",
                            "query": f"retrieval augmented generation evaluation dimension {index}",
                            "rationale": "Covers one confirmed section",
                        }
                        for index in range(1, 4)
                    ]
                }
            )
        if response_model is EvidenceCollection:
            return EvidenceCollection(
                cards=[
                    EvidenceCard(
                        citation_id=f"REF{index:03d}",
                        openalex_id=f"https://openalex.org/W{index}",
                        supported_section_ids=[f"S{index}"],
                        relevance_summary="Directly relevant evidence",
                        key_findings=["The abstract describes an evaluation dimension."],
                    )
                    for index in range(1, 4)
                ]
            )
        raise AssertionError(f"unexpected response model: {response_model}")

    def chat(self, *_: Any, **__: Any) -> str:
        return "# RAG 评测报告\n\n综合证据。[REF001][REF002][REF003]"


class FakeOpenAlexClient:
    def close(self) -> None:
        return None

    def search(self, query: str, *, query_id: str, **_: Any) -> list[ScholarlyWork]:
        index = int(query_id.removeprefix("Q"))
        return [
            ScholarlyWork(
                openalex_id=f"https://openalex.org/W{index}",
                title=query.title(),
                abstract=f"This study evaluates {query} with a reproducible framework.",
                publication_year=2025,
                cited_by_count=20,
                matched_query_ids=[query_id],
            )
        ]


def _initial_state() -> dict[str, Any]:
    request = (
        "调研2023-2026年RAG评测，面向RAG开发者，重点关注检索质量与忠实性，"
        "生成1000字中文标准深度报告，使用3篇论文"
    )
    return {
        "thread_id": "test-thread",
        "mode": "research",
        "original_request": request,
        "enriched_request": request,
        "clarification_round": 0,
        "search_retry_count": 0,
        "status": "started",
    }


def test_followup_mode_routes_directly_to_followup_handler(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "research.db")
    graph = build_research_graph(
        hy3_client=FakeHy3Client(),  # type: ignore[arg-type]
        artifact_store=store,
        openalex_factory=FakeOpenAlexClient,  # type: ignore[arg-type]
        checkpointer=InMemorySaver(),
        followup_handler=lambda thread_id: {
            "answer": f"answer for {thread_id}",
            "metadata": {"grounded": True},
        },
    )
    config = {"configurable": {"thread_id": "followup-thread"}}
    try:
        result = graph.invoke(
            {
                "thread_id": "followup-thread",
                "mode": "followup",
                "followup_question": "有什么局限？",
            },
            config=config,
        )
        assert result["followup_answer"] == "answer for followup-thread"
        assert result["followup_metadata"] == {"grounded": True}
        assert result["status"] == "complete"
        assert "intent" not in result
    finally:
        store.close()


def test_outline_interrupt_does_not_repeat_generation_and_state_stays_compact(tmp_path) -> None:
    fake_hy3 = FakeHy3Client()
    store = ArtifactStore(tmp_path / "research.db")
    graph = build_research_graph(
        hy3_client=fake_hy3,  # type: ignore[arg-type]
        artifact_store=store,
        openalex_factory=FakeOpenAlexClient,  # type: ignore[arg-type]
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test-thread"}}
    try:
        paused = graph.invoke(_initial_state(), config=config)
        assert paused["__interrupt__"][0].value["kind"] == "outline_confirmation"
        assert fake_hy3.outline_calls == 1

        completed = graph.invoke(Command(resume={"action": "approve"}), config=config)

        assert fake_hy3.outline_calls == 1
        assert completed["status"] == "complete"
        assert completed["section_coverage_ratio"] == 1.0
        assert "works" not in completed
        assert "markdown" not in completed
        assert store.get_text("test-thread", "research_report").startswith("# RAG")
    finally:
        store.close()


def test_outline_revision_regenerates_once_then_pauses_again(tmp_path) -> None:
    fake_hy3 = FakeHy3Client()
    store = ArtifactStore(tmp_path / "research.db")
    graph = build_research_graph(
        hy3_client=fake_hy3,  # type: ignore[arg-type]
        artifact_store=store,
        openalex_factory=FakeOpenAlexClient,  # type: ignore[arg-type]
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test-thread"}}
    try:
        graph.invoke(_initial_state(), config=config)
        paused_again = graph.invoke(
            Command(resume={"action": "revise", "feedback": "加强方法比较"}),
            config=config,
        )

        assert paused_again["__interrupt__"][0].value["kind"] == "outline_confirmation"
        assert fake_hy3.outline_calls == 2
    finally:
        store.close()


class AlwaysClarifiesHy3Client(FakeHy3Client):
    def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        if response_model is ResearchIntent:
            return ResearchIntent(
                topic="RAG evaluation",
                goal="compare methods",
                clarification_questions=["请进一步明确核心主题边界"],
            )
        return super().chat_structured(messages, response_model, **kwargs)


def test_clarification_stops_after_three_rounds(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "research.db")
    graph = build_research_graph(
        hy3_client=AlwaysClarifiesHy3Client(),  # type: ignore[arg-type]
        artifact_store=store,
        openalex_factory=FakeOpenAlexClient,  # type: ignore[arg-type]
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "clarification-limit"}}
    try:
        initial_state = _initial_state()
        initial_state.update(
            {
                "original_request": "调研 RAG 评测",
                "enriched_request": "调研 RAG 评测",
            }
        )
        result = graph.invoke(initial_state, config=config)
        for round_number in range(1, 4):
            payload = result["__interrupt__"][0].value
            assert payload["round"] == round_number
            answers = ["RAG 开发者"] * len(payload["questions"])
            result = graph.invoke(Command(resume={"answers": answers}), config=config)

        assert result["status"] == "failed"
        assert result["clarification_round"] == 3
        assert "3 轮" in result["error"]
    finally:
        store.close()


def test_outline_revision_stops_after_five_revisions(tmp_path) -> None:
    fake_hy3 = FakeHy3Client()
    store = ArtifactStore(tmp_path / "research.db")
    graph = build_research_graph(
        hy3_client=fake_hy3,  # type: ignore[arg-type]
        artifact_store=store,
        openalex_factory=FakeOpenAlexClient,  # type: ignore[arg-type]
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "outline-limit"}}
    try:
        result = graph.invoke(_initial_state(), config=config)
        for revision in range(1, 6):
            result = graph.invoke(
                Command(resume={"action": "revise", "feedback": f"修改第 {revision} 次"}),
                config=config,
            )
            assert result["__interrupt__"][0].value["revision_count"] == revision

        result = graph.invoke(
            Command(resume={"action": "revise", "feedback": "再改一次"}),
            config=config,
        )
        assert result["status"] == "failed"
        assert result["outline_revision_count"] == 5
        assert "修改 5 次" in result["error"]
    finally:
        store.close()
