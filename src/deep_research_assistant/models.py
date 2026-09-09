"""Shared data contracts for the research workflow."""

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class ReportDepth(StrEnum):
    """Requested level of detail for the final report."""

    BRIEF = "brief"
    STANDARD = "standard"
    DEEP = "deep"


class ReportLanguage(StrEnum):
    """Supported output languages for the MVP."""

    CHINESE = "zh-CN"
    ENGLISH = "en"


class SourceType(StrEnum):
    """Supported public source categories for literature retrieval."""

    PAPER = "paper"
    SURVEY = "survey"
    DATASET = "dataset"
    OFFICIAL_DOC = "official_doc"


class ResearchIntent(BaseModel):
    """Structured interpretation of a user's research request."""

    topic: str = Field(min_length=2, description="核心调研主题")
    goal: str = Field(min_length=2, description="用户希望通过调研解决的问题")
    audience: str = Field(default="AI 与计算机领域学习者")
    focus_areas: list[str] = Field(default_factory=list)
    excluded_areas: list[str] = Field(default_factory=list)
    start_year: int | None = Field(default=None, ge=1900, le=2100)
    end_year: int | None = Field(default=None, ge=1900, le=2100)
    report_language: ReportLanguage = Field(default=ReportLanguage.CHINESE)
    depth: ReportDepth = Field(default=ReportDepth.STANDARD)
    target_word_count: int | None = Field(default=None, ge=500, le=20000)
    target_source_count: int = Field(default=8, ge=3, le=30)
    source_types: list[SourceType] = Field(default_factory=lambda: [SourceType.PAPER])
    clarification_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_year_range(self) -> "ResearchIntent":
        if self.start_year and self.end_year and self.start_year > self.end_year:
            raise ValueError("start_year 不能晚于 end_year")
        return self


class OutlineSection(BaseModel):
    """A report section and the research questions it must answer."""

    section_id: str = Field(pattern=r"^S\d+$")
    title: str = Field(min_length=2)
    objective: str = Field(min_length=2)
    research_questions: list[str] = Field(min_length=1)
    subsections: list[str] = Field(default_factory=list)


class ResearchOutline(BaseModel):
    """Editable outline produced before literature retrieval."""

    title: str = Field(min_length=2)
    thesis: str = Field(min_length=2, description="报告拟回答的核心问题")
    sections: list[OutlineSection] = Field(min_length=3, max_length=10)


class LiteratureQuery(BaseModel):
    """One executable English query derived from an outline question."""

    query_id: str = Field(pattern=r"^Q\d+$")
    section_id: str = Field(pattern=r"^S\d+$")
    research_question: str = Field(min_length=2)
    query: str = Field(min_length=2, description="用于学术数据库的英文检索式")
    rationale: str = Field(min_length=2)


class LiteratureSearchPlan(BaseModel):
    """Bounded set of queries used to retrieve literature."""

    # Keep the persisted schema backward-compatible with existing M2 artifacts.
    # SearchPlanner bounds newly generated plans to eight queries.
    queries: list[LiteratureQuery] = Field(min_length=3, max_length=12)


class ScholarlyWork(BaseModel):
    """Normalized scholarly work independent of the upstream data source."""

    openalex_id: str
    doi: str | None = None
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    publication_date: str | None = None
    work_type: str | None = None
    source_name: str | None = None
    landing_page_url: str | None = None
    pdf_url: str | None = None
    abstract: str | None = None
    has_full_text: bool = False
    has_pdf: bool = False
    has_grobid_xml: bool = False
    cited_by_count: int = Field(default=0, ge=0)
    is_open_access: bool = False
    topics: list[str] = Field(default_factory=list)
    matched_query_ids: list[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class LiteratureSearchResult(BaseModel):
    """Persisted M2 output with traceable queries and ranked works."""

    intent: ResearchIntent
    outline: ResearchOutline
    search_plan: LiteratureSearchPlan
    retrieved_count: int = Field(ge=0)
    deduplicated_count: int = Field(ge=0)
    works: list[ScholarlyWork] = Field(default_factory=list)


class EvidenceCard(BaseModel):
    """A source-grounded summary used by the report writer."""

    citation_id: str = Field(pattern=r"^REF\d{3}$")
    openalex_id: str
    supported_section_ids: list[str] = Field(default_factory=list)
    relevance_summary: str = Field(min_length=2)
    key_findings: list[str] = Field(min_length=1, max_length=5)
    limitations: list[str] = Field(default_factory=list, max_length=4)


class EvidenceCollection(BaseModel):
    """Validated evidence cards for all selected works."""

    cards: list[EvidenceCard] = Field(min_length=1)
