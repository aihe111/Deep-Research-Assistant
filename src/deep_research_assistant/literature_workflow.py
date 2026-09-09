"""Orchestrate query planning, OpenAlex retrieval, deduplication, and ranking."""

from deep_research_assistant.hy3_client import Hy3Client
from deep_research_assistant.literature_ranker import (
    deduplicate_works,
    rank_works,
    select_with_section_coverage,
)
from deep_research_assistant.models import LiteratureSearchResult
from deep_research_assistant.openalex_client import OpenAlexClient
from deep_research_assistant.search_planner import SearchPlanner
from deep_research_assistant.workflow import ResearchPlan


class LiteratureWorkflow:
    """Execute the complete M2 literature retrieval pipeline."""

    def __init__(
        self,
        hy3_client: Hy3Client | None = None,
        openalex_client: OpenAlexClient | None = None,
    ) -> None:
        self.search_planner = SearchPlanner(hy3_client or Hy3Client())
        self.openalex_client = openalex_client or OpenAlexClient()

    def run(self, plan: ResearchPlan) -> LiteratureSearchResult:
        search_plan = self.search_planner.generate(plan.intent, plan.outline)
        retrieved = []
        try:
            for query in search_plan.queries:
                retrieved.extend(
                    self.openalex_client.search(
                        query.query,
                        query_id=query.query_id,
                        start_year=plan.intent.start_year,
                        end_year=plan.intent.end_year,
                    )
                )
        finally:
            self.openalex_client.close()

        deduplicated = deduplicate_works(retrieved)
        ranked = rank_works(
            deduplicated,
            search_plan.queries,
            start_year=plan.intent.start_year,
            end_year=plan.intent.end_year,
            core_terms=" ".join([plan.intent.topic, *plan.intent.focus_areas]),
        )
        selected = select_with_section_coverage(
            ranked,
            search_plan.queries,
            plan.intent.target_source_count,
        )
        return LiteratureSearchResult(
            intent=plan.intent,
            outline=plan.outline,
            search_plan=search_plan,
            retrieved_count=len(retrieved),
            deduplicated_count=len(deduplicated),
            works=selected,
        )
