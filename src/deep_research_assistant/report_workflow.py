"""Orchestrate M3 evidence building and Markdown report generation."""

from dataclasses import dataclass

from deep_research_assistant.evidence_builder import EvidenceBuilder
from deep_research_assistant.hy3_client import Hy3Client
from deep_research_assistant.models import EvidenceCollection, LiteratureSearchResult
from deep_research_assistant.report_generator import ReportGenerator


@dataclass(frozen=True)
class ReportArtifacts:
    """In-memory M3 artifacts saved by the CLI."""

    evidence: EvidenceCollection
    markdown: str


class ReportWorkflow:
    """Build evidence cards first, then write a citation-validated report."""

    def __init__(self, client: Hy3Client | None = None) -> None:
        model_client = client or Hy3Client()
        self.evidence_builder = EvidenceBuilder(model_client)
        self.report_generator = ReportGenerator(model_client)

    def run(self, search_result: LiteratureSearchResult) -> ReportArtifacts:
        evidence = self.evidence_builder.build(search_result)
        markdown = self.report_generator.generate(search_result, evidence)
        return ReportArtifacts(evidence=evidence, markdown=markdown)
