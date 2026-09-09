"""OpenAlex works API adapter."""

import time
from typing import Any

import httpx

from deep_research_assistant.config import Settings, get_settings
from deep_research_assistant.models import ScholarlyWork


class OpenAlexError(RuntimeError):
    """Raised when OpenAlex cannot return a usable result."""


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Reconstruct plaintext from OpenAlex's positional abstract representation."""

    if not inverted_index:
        return None
    max_position = max(
        (max(positions) for positions in inverted_index.values() if positions), default=-1
    )
    if max_position < 0:
        return None
    words = [""] * (max_position + 1)
    for word, positions in inverted_index.items():
        for position in positions:
            if 0 <= position < len(words):
                words[position] = word
    abstract = " ".join(word for word in words if word).strip()
    return abstract or None


def _parse_work(raw: dict[str, Any], query_id: str) -> ScholarlyWork | None:
    title = (raw.get("display_name") or raw.get("title") or "").strip()
    openalex_id = raw.get("id") or ""
    if not title or not openalex_id:
        return None

    authors = [
        authorship.get("author", {}).get("display_name")
        for authorship in raw.get("authorships") or []
        if authorship.get("author", {}).get("display_name")
    ]
    primary_location = raw.get("primary_location") or {}
    source = primary_location.get("source") or {}
    open_access = raw.get("open_access") or {}
    has_content = raw.get("has_content") or {}
    topics = [
        topic.get("display_name") for topic in raw.get("topics") or [] if topic.get("display_name")
    ][:8]

    return ScholarlyWork(
        openalex_id=openalex_id,
        doi=raw.get("doi"),
        title=title,
        authors=authors,
        publication_year=raw.get("publication_year"),
        publication_date=raw.get("publication_date"),
        work_type=raw.get("type"),
        source_name=source.get("display_name"),
        landing_page_url=primary_location.get("landing_page_url") or raw.get("doi") or openalex_id,
        pdf_url=primary_location.get("pdf_url") or open_access.get("oa_url"),
        abstract=reconstruct_abstract(raw.get("abstract_inverted_index")),
        has_full_text=bool(has_content.get("pdf") or has_content.get("grobid_xml")),
        has_pdf=bool(has_content.get("pdf")),
        has_grobid_xml=bool(has_content.get("grobid_xml")),
        cited_by_count=max(int(raw.get("cited_by_count") or 0), 0),
        is_open_access=bool(open_access.get("is_oa")),
        topics=topics,
        matched_query_ids=[query_id],
    )


class OpenAlexClient:
    """Search and normalize scholarly works from OpenAlex."""

    SELECT_FIELDS = ",".join(
        [
            "id",
            "doi",
            "title",
            "display_name",
            "publication_year",
            "publication_date",
            "type",
            "authorships",
            "primary_location",
            "abstract_inverted_index",
            "has_content",
            "cited_by_count",
            "open_access",
            "topics",
        ]
    )

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.settings.openalex_base_url,
            timeout=30.0,
            headers={"User-Agent": "Deep-Research-Assistant/0.1"},
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def search(
        self,
        query: str,
        *,
        query_id: str,
        start_year: int | None,
        end_year: int | None,
        per_page: int | None = None,
    ) -> list[ScholarlyWork]:
        filters = ["is_retracted:false", "has_abstract:true"]
        if start_year:
            filters.append(f"from_publication_date:{start_year}-01-01")
        if end_year:
            filters.append(f"to_publication_date:{end_year}-12-31")

        params: dict[str, str | int] = {
            "search": query,
            "filter": ",".join(filters),
            "sort": "relevance_score:desc",
            "per_page": per_page or self.settings.openalex_per_query,
            "select": self.SELECT_FIELDS,
        }
        if self.settings.openalex_api_key:
            params["api_key"] = self.settings.openalex_api_key
        if self.settings.openalex_email:
            params["mailto"] = self.settings.openalex_email

        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                response = self._client.get("/works", params=params)
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise OpenAlexError(f"OpenAlex 网络请求失败：{exc}") from exc
                time.sleep(0.5 * (2**attempt))
                continue
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))

        if response is None:
            raise OpenAlexError("OpenAlex 未返回响应")
        if response.status_code in {401, 403}:
            raise OpenAlexError("OpenAlex 拒绝访问，请检查 OPENALEX_API_KEY")
        try:
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            detail = response.text[:300]
            raise OpenAlexError(f"OpenAlex 响应无效：{detail}") from exc

        works: list[ScholarlyWork] = []
        for raw in payload.get("results") or []:
            work = _parse_work(raw, query_id)
            if work:
                works.append(work)
        return works
