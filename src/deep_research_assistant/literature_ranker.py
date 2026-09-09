"""Deterministic deduplication and explainable local relevance ranking."""

import math
import re
import unicodedata

from deep_research_assistant.models import LiteratureQuery, ScholarlyWork

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _tokens(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(text.casefold()) if token not in STOP_WORDS}


def deduplicate_works(works: list[ScholarlyWork]) -> list[ScholarlyWork]:
    """Merge exact-title duplicates and preserve every matched query ID."""

    merged: dict[str, ScholarlyWork] = {}
    for work in works:
        key = _normalize_title(work.title) or work.openalex_id
        current = merged.get(key)
        if current is None:
            merged[key] = work
            continue

        preferred = max(
            (current, work),
            key=lambda item: (
                bool(item.doi),
                bool(item.abstract),
                item.cited_by_count,
                bool(item.pdf_url),
            ),
        )
        query_ids = sorted(set(current.matched_query_ids + work.matched_query_ids))
        updates: dict[str, object] = {"matched_query_ids": query_ids}
        if not preferred.abstract:
            updates["abstract"] = current.abstract or work.abstract
        if not preferred.doi:
            updates["doi"] = current.doi or work.doi
        merged[key] = preferred.model_copy(update=updates)
    return list(merged.values())


def _term_recall(query_tokens: set[str], text_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def rank_works(
    works: list[ScholarlyWork],
    queries: list[LiteratureQuery],
    *,
    start_year: int | None,
    end_year: int | None,
    core_terms: str = "",
) -> list[ScholarlyWork]:
    """Rank works using lexical match, core-topic match, citations, and recency."""

    query_by_id = {query.query_id: _tokens(query.query) for query in queries}
    core_tokens = _tokens(core_terms)
    ranked: list[ScholarlyWork] = []
    for work in works:
        matched_tokens = [query_by_id[qid] for qid in work.matched_query_ids if qid in query_by_id]
        if not matched_tokens:
            matched_tokens = list(query_by_id.values())

        title_tokens = _tokens(work.title)
        abstract_tokens = _tokens(work.abstract or "")
        title_score = max(
            (_term_recall(query, title_tokens) for query in matched_tokens), default=0.0
        )
        abstract_score = max(
            (_term_recall(query, abstract_tokens) for query in matched_tokens), default=0.0
        )
        core_score = max(
            _term_recall(core_tokens, title_tokens),
            0.6 * _term_recall(core_tokens, abstract_tokens),
        )
        coverage_score = min(len(set(work.matched_query_ids)) / 3, 1.0)
        citation_score = min(math.log1p(work.cited_by_count) / math.log1p(500), 1.0)

        recency_score = 0.5
        if work.publication_year and start_year and end_year and end_year > start_year:
            recency_score = min(
                max((work.publication_year - start_year) / (end_year - start_year), 0), 1
            )

        score = (
            0.42 * title_score
            + 0.28 * abstract_score
            + 0.15 * core_score
            + 0.05 * coverage_score
            + 0.05 * citation_score
            + 0.05 * recency_score
        )
        ranked.append(work.model_copy(update={"relevance_score": round(score, 4)}))

    return sorted(
        ranked,
        key=lambda item: (item.relevance_score, item.cited_by_count, item.publication_year or 0),
        reverse=True,
    )


def select_with_section_coverage(
    ranked: list[ScholarlyWork],
    queries: list[LiteratureQuery],
    limit: int,
) -> list[ScholarlyWork]:
    """Select top works while reserving one candidate for each searchable section."""

    query_section = {query.query_id: query.section_id for query in queries}
    ordered_sections = list(dict.fromkeys(query.section_id for query in queries))
    selected: list[ScholarlyWork] = []
    selected_ids: set[str] = set()

    for section_id in ordered_sections:
        candidate = next(
            (
                work
                for work in ranked
                if work.openalex_id not in selected_ids
                and section_id
                in {
                    query_section[query_id]
                    for query_id in work.matched_query_ids
                    if query_id in query_section
                }
            ),
            None,
        )
        if candidate is not None and len(selected) < limit:
            selected.append(candidate)
            selected_ids.add(candidate.openalex_id)

    for work in ranked:
        if len(selected) >= limit:
            break
        if work.openalex_id not in selected_ids:
            selected.append(work)
            selected_ids.add(work.openalex_id)

    return sorted(
        selected,
        key=lambda item: (item.relevance_score, item.cited_by_count, item.publication_year or 0),
        reverse=True,
    )


def section_coverage(
    works: list[ScholarlyWork],
    queries: list[LiteratureQuery],
) -> tuple[int, int]:
    """Return covered and total searchable outline sections."""

    query_section = {query.query_id: query.section_id for query in queries}
    expected = set(query_section.values())
    covered = {
        query_section[query_id]
        for work in works
        for query_id in work.matched_query_ids
        if query_id in query_section
    }
    return len(covered), len(expected)
