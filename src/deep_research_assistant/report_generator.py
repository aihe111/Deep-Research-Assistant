"""Generate a grounded Markdown report and deterministic bibliography."""

import json
import re

from deep_research_assistant.hy3_client import Hy3Client
from deep_research_assistant.models import EvidenceCollection, LiteratureSearchResult, ScholarlyWork

CITATION_PATTERN = re.compile(r"\[(REF\d{3})\]")

SYSTEM_PROMPT = """你是调研 Agent 的报告写作节点。请根据已确认的大纲和证据卡生成中文 Markdown
调研报告正文。

规则：
1. 只能使用输入证据卡中的信息，不得凭模型记忆增加论文、数字或结论。
2. 可核查的论文观点必须在句末使用 [REF001] 格式引用；可以连续引用多个来源。
3. citation_id 只能使用输入提供的编号，禁止编造编号。
4. 区分“文献明确指出”和“基于多篇证据的综合判断”，证据不足时明确说明。
5. 按输入大纲组织一级和二级标题，避免逐篇罗列摘要，要比较和综合不同来源。
6. 报告开头包含标题和简短摘要，结尾包含局限与结论。
7. 以 intent.target_word_count 为目标字数，允许上下浮动约 20%；若该字段为空则采用适中篇幅。
8. 不生成“参考文献”章节，程序会根据实际引用确定性追加。
9. 直接输出 Markdown，不使用围栏代码块，也不解释写作过程。
"""


class ReportCitationError(RuntimeError):
    """Raised when report citations cannot be mapped to retrieved works."""


def extract_citation_ids(markdown: str) -> list[str]:
    """Return cited IDs in first-appearance order."""

    return list(dict.fromkeys(CITATION_PATTERN.findall(markdown)))


def validate_citations(markdown: str, allowed_ids: set[str]) -> list[str]:
    citation_ids = extract_citation_ids(markdown)
    if not citation_ids:
        raise ReportCitationError("报告正文没有引用任何证据来源")
    unknown = set(citation_ids) - allowed_ids
    if unknown:
        raise ReportCitationError(f"报告包含未知引用：{sorted(unknown)}")
    return citation_ids


def _render_reference(citation_id: str, work: ScholarlyWork) -> str:
    authors = ", ".join(work.authors[:6]) or "Unknown author"
    if len(work.authors) > 6:
        authors += ", et al."
    year = str(work.publication_year or "n.d.")
    source = f" {work.source_name}." if work.source_name else ""
    url = work.doi or work.landing_page_url or work.openalex_id
    return f"- [{citation_id}] {authors}. ({year}). {work.title}.{source} {url}"


def append_deterministic_references(
    markdown: str,
    search_result: LiteratureSearchResult,
) -> str:
    """Validate citations and append metadata-backed references only for cited works."""

    work_by_citation = {
        f"REF{index:03d}": work for index, work in enumerate(search_result.works, start=1)
    }
    citation_ids = sorted(
        validate_citations(markdown, set(work_by_citation)),
        key=lambda citation_id: int(citation_id.removeprefix("REF")),
    )
    references = [
        _render_reference(citation_id, work_by_citation[citation_id])
        for citation_id in citation_ids
    ]
    return markdown.rstrip() + "\n\n## 参考文献\n\n" + "\n".join(references) + "\n"


class ReportGenerator:
    """Write a report using only validated evidence cards."""

    def __init__(self, client: Hy3Client) -> None:
        self.client = client

    def generate_body(
        self,
        search_result: LiteratureSearchResult,
        evidence: EvidenceCollection,
    ) -> str:
        payload = json.dumps(
            {
                "intent": search_result.intent.model_dump(mode="json"),
                "outline": search_result.outline.model_dump(mode="json"),
                "evidence_cards": evidence.model_dump(mode="json")["cards"],
            },
            ensure_ascii=False,
        )
        target_words = search_result.intent.target_word_count or 4_000
        output_tokens = min(24_000, max(2_000, int(target_words * 1.4) + 500))
        return self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0.2,
            reasoning_effort="low",
            stage="report",
            max_output_tokens=output_tokens,
        )

    def generate(
        self,
        search_result: LiteratureSearchResult,
        evidence: EvidenceCollection,
    ) -> str:
        body = self.generate_body(search_result, evidence)
        return append_deterministic_references(body, search_result)
