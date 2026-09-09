"""Create grounded evidence cards from selected OpenAlex metadata and abstracts."""

import json

from deep_research_assistant.hy3_client import Hy3Client
from deep_research_assistant.llm_policy import estimate_text_tokens
from deep_research_assistant.models import EvidenceCollection, LiteratureSearchResult

ABSTRACT_TOTAL_TOKEN_BUDGET = 18_000
ABSTRACT_PER_SOURCE_MAX_TOKENS = 2_000

SYSTEM_PROMPT = """你是调研 Agent 的证据整理节点。你只能使用输入中的标题、摘要和元数据，
不能引入模型记忆、猜测论文正文或虚构实验数字。

规则：
1. 为每个输入来源生成且只生成一张证据卡，citation_id 必须原样保留。
2. relevance_summary 说明该来源为什么与本次调研相关。
3. key_findings 只能概括摘要明确支持的内容；摘要未给出数值时不得补充数值。
4. limitations 记录摘要缺失的信息、研究范围限制或仅凭摘要无法确认之处。
5. supported_section_ids 只能从输入大纲的 section_id 中选择，可为空。
6. 使用中文概括，但保留必要的英文术语。
7. 不复制长段摘要，不生成引用列表。
8. 输出必须符合指定 JSON Schema。
"""


class EvidenceBuildError(RuntimeError):
    """Raised when generated evidence loses source traceability."""


class EvidenceBuilder:
    """Build and validate one traceable evidence card per selected work."""

    def __init__(self, client: Hy3Client) -> None:
        self.client = client

    def build(self, search_result: LiteratureSearchResult) -> EvidenceCollection:
        valid_sections = {section.section_id for section in search_result.outline.sections}
        source_by_citation = {
            f"REF{index:03d}": work for index, work in enumerate(search_result.works, start=1)
        }
        abstract_budget = min(
            ABSTRACT_PER_SOURCE_MAX_TOKENS,
            max(300, ABSTRACT_TOTAL_TOKEN_BUDGET // max(len(source_by_citation), 1)),
        )
        source_payload = [
            {
                "citation_id": citation_id,
                "openalex_id": work.openalex_id,
                "title": work.title,
                "authors": work.authors,
                "publication_year": work.publication_year,
                "source_name": work.source_name,
                "abstract": _bounded_abstract(work.abstract, abstract_budget),
                "matched_query_ids": work.matched_query_ids,
            }
            for citation_id, work in source_by_citation.items()
        ]
        outline_payload = [
            {
                "section_id": section.section_id,
                "title": section.title,
                "objective": section.objective,
            }
            for section in search_result.outline.sections
        ]
        user_payload = json.dumps(
            {
                "research_goal": search_result.intent.goal,
                "focus_areas": search_result.intent.focus_areas,
                "outline": outline_payload,
                "sources": source_payload,
            },
            ensure_ascii=False,
        )
        collection = self.client.chat_structured(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            EvidenceCollection,
            schema_name="evidence_collection",
            reasoning_effort="no_think",
            stage="evidence",
        )

        actual_ids = [card.citation_id for card in collection.cards]
        expected_ids = list(source_by_citation)
        if len(actual_ids) != len(set(actual_ids)):
            raise EvidenceBuildError("证据卡包含重复 citation_id")
        if set(actual_ids) != set(expected_ids):
            raise EvidenceBuildError(
                f"证据卡来源不完整：expected={expected_ids}, actual={actual_ids}"
            )

        validated_cards = []
        for card in collection.cards:
            invalid_sections = set(card.supported_section_ids) - valid_sections
            if invalid_sections:
                raise EvidenceBuildError(f"证据卡引用了不存在的章节：{sorted(invalid_sections)}")
            source = source_by_citation[card.citation_id]
            validated_cards.append(card.model_copy(update={"openalex_id": source.openalex_id}))
        validated_cards.sort(key=lambda card: card.citation_id)
        return EvidenceCollection(cards=validated_cards)


def _bounded_abstract(abstract: str | None, max_tokens: int) -> str | None:
    """Fit each abstract into its fair share of the evidence-stage input budget."""

    if not abstract:
        return None
    if estimate_text_tokens(abstract) <= max_tokens:
        return abstract
    content_budget = max(max_tokens - estimate_text_tokens("…"), 0)
    low, high = 0, len(abstract)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_text_tokens(abstract[:middle]) <= content_budget:
            low = middle
        else:
            high = middle - 1
    return abstract[:low].rstrip() + "…"
