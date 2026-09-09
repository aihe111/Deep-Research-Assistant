"""LangGraph workflow for the end-to-end research assistant."""

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from deep_research_assistant.artifact_store import ArtifactStore
from deep_research_assistant.evidence_builder import EvidenceBuilder
from deep_research_assistant.hy3_client import Hy3Client
from deep_research_assistant.intent_analyzer import IntentAnalyzer
from deep_research_assistant.literature_ranker import (
    deduplicate_works,
    rank_works,
    section_coverage,
    select_with_section_coverage,
)
from deep_research_assistant.models import (
    EvidenceCollection,
    LiteratureSearchPlan,
    LiteratureSearchResult,
    ResearchIntent,
    ResearchOutline,
    ScholarlyWork,
)
from deep_research_assistant.openalex_client import OpenAlexClient
from deep_research_assistant.outline_generator import OutlineGenerator
from deep_research_assistant.report_generator import ReportGenerator
from deep_research_assistant.search_planner import SearchPlanner

MAX_SEARCH_RETRIES = 2
MAX_CLARIFICATION_ROUNDS = 3
MAX_OUTLINE_REVISIONS = 5


class ResearchGraphState(TypedDict, total=False):
    """Compact checkpoint state; large content is stored in ArtifactStore."""

    thread_id: str
    mode: Literal["research", "followup"]
    followup_question: str
    followup_answer: str
    followup_metadata: dict[str, Any]
    original_request: str
    enriched_request: str
    intent: dict[str, Any]
    clarification_questions: list[str]
    clarification_round: int
    outline: dict[str, Any]
    outline_feedback: str
    outline_approved: bool
    outline_revision_count: int
    search_plan: dict[str, Any]
    search_feedback: str
    search_retry_count: int
    raw_works_artifact: str
    literature_artifact: str
    evidence_artifact: str
    report_artifact: str
    retrieved_count: int
    deduplicated_count: int
    selected_count: int
    covered_section_count: int
    searchable_section_count: int
    section_coverage_ratio: float
    average_relevance: float
    status: str
    error: str


class ResearchGraphNodes:
    """Node implementations with dependencies injected for testability."""

    def __init__(
        self,
        hy3_client: Hy3Client,
        artifact_store: ArtifactStore,
        openalex_factory: Callable[[], OpenAlexClient],
        followup_handler: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.intent_analyzer = IntentAnalyzer(hy3_client)
        self.outline_generator = OutlineGenerator(hy3_client)
        self.search_planner = SearchPlanner(hy3_client)
        self.evidence_builder = EvidenceBuilder(hy3_client)
        self.report_generator = ReportGenerator(hy3_client)
        self.artifacts = artifact_store
        self.openalex_factory = openalex_factory
        self.followup_handler = followup_handler

    def answer_follow_up(self, state: ResearchGraphState) -> dict[str, Any]:
        """Answer a report follow-up as a separate LangGraph Run."""

        if self.followup_handler is None:
            raise RuntimeError("当前图没有配置报告追问处理器")
        result = self.followup_handler(state["thread_id"])
        return {
            "mode": "followup",
            "followup_answer": str(result["answer"]),
            "followup_metadata": dict(result.get("metadata") or {}),
            "status": "complete",
            "error": "",
        }

    def analyze_intent(self, state: ResearchGraphState) -> dict[str, Any]:
        request = state.get("enriched_request") or state["original_request"]
        intent = self.intent_analyzer.analyze(request)
        clarification_limit_reached = bool(
            intent.clarification_questions
            and state.get("clarification_round", 0) >= MAX_CLARIFICATION_ROUNDS
        )
        return {
            "enriched_request": request,
            "intent": intent.model_dump(mode="json"),
            "clarification_questions": intent.clarification_questions,
            "status": (
                "failed"
                if clarification_limit_reached
                else "waiting_for_clarification"
                if intent.clarification_questions
                else "intent_confirmed"
            ),
            "error": (
                "经过 3 轮补充后调研意图仍不完整，请新建会话并更明确地描述主题、目标、"
                "报告字数、时间范围和文献数量。"
                if clarification_limit_reached
                else ""
            ),
        }

    @staticmethod
    def clarify_intent(state: ResearchGraphState) -> dict[str, Any]:
        """Pause in a side-effect-free node so resume cannot repeat an API call."""

        response = interrupt(
            {
                "kind": "clarification",
                "questions": state["clarification_questions"],
                "round": state.get("clarification_round", 0) + 1,
                "max_rounds": MAX_CLARIFICATION_ROUNDS,
            }
        )
        if not isinstance(response, dict) or not isinstance(response.get("answers"), list):
            raise ValueError("补充信息格式无效，应包含 answers 列表")
        answers = [str(answer).strip() for answer in response["answers"]]
        questions = state["clarification_questions"]
        if len(answers) != len(questions) or any(not answer for answer in answers):
            raise ValueError("每个澄清问题都必须提供非空回答")
        supplement = "\n".join(
            f"- {question} 回答：{answer}"
            for question, answer in zip(questions, answers, strict=True)
        )
        enriched = state.get("enriched_request") or state["original_request"]
        return {
            "enriched_request": enriched + "\n\n用户补充信息：\n" + supplement,
            "clarification_questions": [],
            "clarification_round": state.get("clarification_round", 0) + 1,
            "status": "clarification_received",
        }

    def generate_outline(self, state: ResearchGraphState) -> dict[str, Any]:
        intent = ResearchIntent.model_validate(state["intent"])
        outline = self.outline_generator.generate(intent, state.get("outline_feedback") or None)
        return {
            "outline": outline.model_dump(mode="json"),
            "outline_approved": False,
            "status": "waiting_for_outline_confirmation",
        }

    @staticmethod
    def confirm_outline(state: ResearchGraphState) -> dict[str, Any]:
        """Pause after outline generation without repeating generation on resume."""

        response = interrupt(
            {
                "kind": "outline_confirmation",
                "outline": state["outline"],
                "message": "请确认大纲；也可以给出修改意见后重新生成。",
                "revision_count": state.get("outline_revision_count", 0),
                "max_revisions": MAX_OUTLINE_REVISIONS,
            }
        )
        if not isinstance(response, dict):
            raise ValueError("大纲确认格式无效")
        action = str(response.get("action", "")).strip().casefold()
        feedback = str(response.get("feedback", "")).strip()
        if action not in {"approve", "revise"}:
            raise ValueError("大纲操作必须是 approve 或 revise")
        if action == "revise" and not feedback:
            raise ValueError("修改大纲时必须提供 feedback")
        revision_count = state.get("outline_revision_count", 0)
        if action == "revise" and revision_count >= MAX_OUTLINE_REVISIONS:
            return {
                "outline_approved": False,
                "status": "failed",
                "error": "大纲已经修改 5 次，仍未确认。请新建会话并在初始需求中明确大纲要求。",
            }
        return {
            "outline_approved": action == "approve",
            "outline_feedback": feedback,
            "outline_revision_count": revision_count + (1 if action == "revise" else 0),
            "status": "outline_confirmed" if action == "approve" else "outline_revision_requested",
        }

    def plan_search(self, state: ResearchGraphState) -> dict[str, Any]:
        intent = ResearchIntent.model_validate(state["intent"])
        outline = ResearchOutline.model_validate(state["outline"])
        plan = self.search_planner.generate(
            intent,
            outline,
            state.get("search_feedback") or None,
        )
        return {
            "search_plan": plan.model_dump(mode="json"),
            "status": "search_planned",
        }

    def retrieve_literature(self, state: ResearchGraphState) -> dict[str, Any]:
        intent = ResearchIntent.model_validate(state["intent"])
        plan = LiteratureSearchPlan.model_validate(state["search_plan"])
        retrieved: list[ScholarlyWork] = []
        client = self.openalex_factory()
        try:
            for query in plan.queries:
                retrieved.extend(
                    client.search(
                        query.query,
                        query_id=query.query_id,
                        start_year=intent.start_year,
                        end_year=intent.end_year,
                    )
                )
        finally:
            client.close()

        artifact_key = "raw_works"
        self.artifacts.put_json(
            state["thread_id"],
            artifact_key,
            [work.model_dump(mode="json") for work in retrieved],
        )
        return {
            "raw_works_artifact": artifact_key,
            "retrieved_count": len(retrieved),
            "status": "literature_retrieved",
        }

    def process_literature(self, state: ResearchGraphState) -> dict[str, Any]:
        intent = ResearchIntent.model_validate(state["intent"])
        outline = ResearchOutline.model_validate(state["outline"])
        plan = LiteratureSearchPlan.model_validate(state["search_plan"])
        retrieved = [
            ScholarlyWork.model_validate(item)
            for item in self.artifacts.get_json(state["thread_id"], state["raw_works_artifact"])
        ]
        deduplicated = deduplicate_works(retrieved)
        ranked = rank_works(
            deduplicated,
            plan.queries,
            start_year=intent.start_year,
            end_year=intent.end_year,
            core_terms=" ".join([intent.topic, *intent.focus_areas]),
        )
        selected = select_with_section_coverage(
            ranked,
            plan.queries,
            intent.target_source_count,
        )
        covered, total_sections = section_coverage(selected, plan.queries)
        coverage_ratio = covered / total_sections if total_sections else 1.0
        average_relevance = (
            sum(work.relevance_score for work in selected) / len(selected) if selected else 0.0
        )

        minimum_count = min(intent.target_source_count, max(3, total_sections))
        quality_problems: list[str] = []
        if len(selected) < minimum_count:
            quality_problems.append(f"有效文献仅 {len(selected)} 篇，期望至少 {minimum_count} 篇")
        if coverage_ratio < 0.75:
            quality_problems.append(
                f"检索章节覆盖率仅 {coverage_ratio:.0%}（{covered}/{total_sections}）"
            )
        if selected and average_relevance < 0.18:
            quality_problems.append(f"平均相关性分数偏低（{average_relevance:.3f}）")

        retry_count = state.get("search_retry_count", 0)
        if quality_problems and retry_count < MAX_SEARCH_RETRIES:
            return {
                "deduplicated_count": len(deduplicated),
                "selected_count": len(selected),
                "covered_section_count": covered,
                "searchable_section_count": total_sections,
                "section_coverage_ratio": round(coverage_ratio, 4),
                "average_relevance": round(average_relevance, 4),
                "search_feedback": "；".join(quality_problems),
                "search_retry_count": retry_count + 1,
                "status": "search_retry_needed",
            }

        if not selected:
            return {
                "deduplicated_count": len(deduplicated),
                "selected_count": 0,
                "search_feedback": "；".join(quality_problems) or "未检索到可用文献",
                "status": "failed",
                "error": "多轮检索后仍没有可用文献，请调整主题、年份或关键词。",
            }

        result = LiteratureSearchResult(
            intent=intent,
            outline=outline,
            search_plan=plan,
            retrieved_count=state["retrieved_count"],
            deduplicated_count=len(deduplicated),
            works=selected,
        )
        artifact_key = "literature_result"
        self.artifacts.put_json(
            state["thread_id"],
            artifact_key,
            result.model_dump(mode="json"),
        )
        return {
            "literature_artifact": artifact_key,
            "deduplicated_count": len(deduplicated),
            "selected_count": len(selected),
            "covered_section_count": covered,
            "searchable_section_count": total_sections,
            "section_coverage_ratio": round(coverage_ratio, 4),
            "average_relevance": round(average_relevance, 4),
            "search_feedback": "；".join(quality_problems),
            "status": "literature_ready",
        }

    def build_evidence(self, state: ResearchGraphState) -> dict[str, Any]:
        result = LiteratureSearchResult.model_validate(
            self.artifacts.get_json(state["thread_id"], state["literature_artifact"])
        )
        evidence = self.evidence_builder.build(result)
        artifact_key = "evidence_cards"
        self.artifacts.put_json(
            state["thread_id"],
            artifact_key,
            evidence.model_dump(mode="json"),
        )
        return {"evidence_artifact": artifact_key, "status": "evidence_ready"}

    def generate_report(self, state: ResearchGraphState) -> dict[str, Any]:
        result = LiteratureSearchResult.model_validate(
            self.artifacts.get_json(state["thread_id"], state["literature_artifact"])
        )
        evidence = EvidenceCollection.model_validate(
            self.artifacts.get_json(state["thread_id"], state["evidence_artifact"])
        )
        markdown = self.report_generator.generate(result, evidence)
        artifact_key = "research_report"
        self.artifacts.put_text(
            state["thread_id"],
            artifact_key,
            markdown,
            content_type="text/markdown",
        )
        return {
            "report_artifact": artifact_key,
            "status": "complete",
            "error": "",
        }


def _route_after_intent(
    state: ResearchGraphState,
) -> Literal["clarify_intent", "generate_outline", "end"]:
    if state.get("status") == "failed":
        return "end"
    return "clarify_intent" if state.get("clarification_questions") else "generate_outline"


def _route_from_start(
    state: ResearchGraphState,
) -> Literal["answer_follow_up", "analyze_intent"]:
    return "answer_follow_up" if state.get("mode") == "followup" else "analyze_intent"


def _route_after_outline(
    state: ResearchGraphState,
) -> Literal["plan_search", "generate_outline", "end"]:
    if state.get("status") == "failed":
        return "end"
    return "plan_search" if state.get("outline_approved") else "generate_outline"


def _route_after_processing(
    state: ResearchGraphState,
) -> Literal["plan_search", "build_evidence", "end"]:
    if state.get("status") == "search_retry_needed":
        return "plan_search"
    if state.get("status") == "failed":
        return "end"
    return "build_evidence"


def build_research_graph(
    *,
    hy3_client: Hy3Client,
    artifact_store: ArtifactStore,
    openalex_factory: Callable[[], OpenAlexClient],
    checkpointer: Any = None,
    followup_handler: Callable[[str], dict[str, Any]] | None = None,
) -> Any:
    """Build and compile the production workflow."""

    nodes = ResearchGraphNodes(
        hy3_client,
        artifact_store,
        openalex_factory,
        followup_handler,
    )
    graph = StateGraph(ResearchGraphState)
    graph.add_node("answer_follow_up", nodes.answer_follow_up)
    graph.add_node("analyze_intent", nodes.analyze_intent)
    graph.add_node("clarify_intent", nodes.clarify_intent)
    graph.add_node("generate_outline", nodes.generate_outline)
    graph.add_node("confirm_outline", nodes.confirm_outline)
    graph.add_node("plan_search", nodes.plan_search)
    graph.add_node("retrieve_literature", nodes.retrieve_literature)
    graph.add_node("process_literature", nodes.process_literature)
    graph.add_node("build_evidence", nodes.build_evidence)
    graph.add_node("generate_report", nodes.generate_report)

    graph.add_conditional_edges(
        START,
        _route_from_start,
        {
            "answer_follow_up": "answer_follow_up",
            "analyze_intent": "analyze_intent",
        },
    )
    graph.add_edge("answer_follow_up", END)
    graph.add_conditional_edges(
        "analyze_intent",
        _route_after_intent,
        {
            "clarify_intent": "clarify_intent",
            "generate_outline": "generate_outline",
            "end": END,
        },
    )
    graph.add_edge("clarify_intent", "analyze_intent")
    graph.add_edge("generate_outline", "confirm_outline")
    graph.add_conditional_edges(
        "confirm_outline",
        _route_after_outline,
        {"plan_search": "plan_search", "generate_outline": "generate_outline", "end": END},
    )
    graph.add_edge("plan_search", "retrieve_literature")
    graph.add_edge("retrieve_literature", "process_literature")
    graph.add_conditional_edges(
        "process_literature",
        _route_after_processing,
        {"plan_search": "plan_search", "build_evidence": "build_evidence", "end": END},
    )
    graph.add_edge("build_evidence", "generate_report")
    graph.add_edge("generate_report", END)
    return graph.compile(checkpointer=checkpointer)
