"""带来源等级的研究工具，以及运行时动态 MCP 装载。"""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field

from deep_research_assistant.config import Settings
from deep_research_assistant.deep_research_model import build_chat_model
from deep_research_assistant.deep_research_prompts import WEBPAGE_SUMMARY_PROMPT
from deep_research_assistant.deep_research_state import SourceRecord, WebpageSummary
from deep_research_assistant.hy3_budget import Hy3CallBudgetManager
from deep_research_assistant.models import ScholarlyWork
from deep_research_assistant.openalex_client import OpenAlexClient
from deep_research_assistant.scholarly_fulltext import (
    OpenAlexFullTextClient,
    ScholarlyFullTextError,
    normalize_openalex_id,
)


class OpenAlexSearchInput(BaseModel):
    query: str = Field(min_length=2, description="用于学术数据库的检索式，通常优先使用英文。")
    start_year: int | None = Field(default=None, ge=1900, le=2100)
    end_year: int | None = Field(default=None, ge=1900, le=2100)
    max_results: int = Field(default=10, ge=1, le=50)


class OpenAlexFullTextInput(BaseModel):
    openalex_ids: list[str] = Field(
        min_length=1,
        max_length=10,
        description="需要读取正文的 OpenAlex Work ID，例如 W3038568908。",
    )


class TavilySearchInput(BaseModel):
    queries: list[str] = Field(
        min_length=1,
        max_length=10,
        description="一组可以并发执行、彼此聚焦的网页检索式。",
    )


def _canonical_url(value: str) -> str:
    """Normalize common tracking variants without changing meaningful query parameters."""

    parts = urlsplit(value.strip())
    kept_query = [
        (key, item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/") or "/",
            urlencode(kept_query),
            "",
        )
    )


class TavilySummaryCache:
    """Bounded process-local URL cache with per-URL locks for parallel researchers."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max_entries
        self._values: OrderedDict[str, SourceRecord] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def lock_for(self, url: str) -> asyncio.Lock:
        async with self._guard:
            return self._locks.setdefault(url, asyncio.Lock())

    async def get(self, url: str) -> SourceRecord | None:
        async with self._guard:
            record = self._values.get(url)
            if record is None:
                return None
            self._values.move_to_end(url)
            return record.model_copy(deep=True)

    async def put(self, url: str, record: SourceRecord) -> None:
        async with self._guard:
            self._values[url] = record.model_copy(deep=True)
            self._values.move_to_end(url)
            while len(self._values) > self.max_entries:
                stale_url, _ = self._values.popitem(last=False)
                self._locks.pop(stale_url, None)


@tool
def think_tool(reflection: str) -> str:
    """反思检索策略、证据质量、相互矛盾之处以及仍待补足的信息。"""

    return f"已记录反思：{reflection}"


def _work_key(value: str) -> str:
    try:
        return normalize_openalex_id(value)
    except ScholarlyFullTextError:
        return value


def create_openalex_tool(
    settings: Settings,
    work_registry: dict[str, ScholarlyWork] | None = None,
) -> BaseTool:
    """把现有 OpenAlex 元数据与摘要检索封装成异步 LangChain Tool。"""

    registry = work_registry if work_registry is not None else {}

    async def search(
        query: str,
        start_year: int | None = None,
        end_year: int | None = None,
        max_results: int = 10,
    ) -> str:
        def run() -> list[SourceRecord]:
            client = OpenAlexClient(settings)
            try:
                works = client.search(
                    query,
                    query_id="deep-research",
                    start_year=start_year,
                    end_year=end_year,
                    per_page=max_results,
                )
            finally:
                client.close()
            for work in works:
                registry[_work_key(work.openalex_id)] = work
            return [
                SourceRecord(
                    provider="openalex",
                    source_type="scholarly_work",
                    evidence_level="abstract",
                    title=work.title,
                    url=work.doi or work.landing_page_url or work.openalex_id,
                    content=work.abstract or "",
                    published_at=work.publication_date
                    or (str(work.publication_year) if work.publication_year else None),
                    authors=work.authors,
                    identifiers={
                        key: value
                        for key, value in {
                            "openalex_id": work.openalex_id,
                            "doi": work.doi,
                        }.items()
                        if value
                    },
                    metadata={
                        "venue": work.source_name,
                        "work_type": work.work_type,
                        "cited_by_count": work.cited_by_count,
                        "is_open_access": work.is_open_access,
                        "pdf_url": work.pdf_url,
                        "has_full_text": work.has_full_text,
                        "has_pdf": work.has_pdf,
                        "has_grobid_xml": work.has_grobid_xml,
                        "topics": work.topics,
                    },
                )
                for work in works
            ]

        records = await asyncio.to_thread(run)
        return json.dumps(
            {
                "provider": "openalex",
                "evidence_level": "abstract",
                "citation_rule": (
                    "优先使用 DOI；没有 DOI 时使用 OpenAlex 或落地页，并保留作者与年份。"
                    "需要引用方法、实验、结果或局限时，应继续调用 openalex_fetch_fulltext。"
                ),
                "results": [record.model_dump(exclude_none=True) for record in records],
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        coroutine=search,
        name="openalex_search",
        description=(
            "在 OpenAlex 中检索学术成果的元数据和摘要，适合发现论文、综述和学术证据。"
            "结果会标明是否存在 OpenAlex 全文；摘要不能替代方法、实验和局限等正文证据。"
        ),
        args_schema=OpenAlexSearchInput,
    )


def create_openalex_fulltext_tool(
    settings: Settings,
    work_registry: dict[str, ScholarlyWork] | None = None,
    client_factory: Callable[[Settings], OpenAlexFullTextClient] | None = None,
) -> BaseTool:
    """创建按 OpenAlex Work ID 读取官方 GROBID XML/PDF 正文的工具。"""

    registry = work_registry if work_registry is not None else {}
    make_client = client_factory or OpenAlexFullTextClient

    async def fetch_fulltext(openalex_ids: list[str]) -> str:
        unique_ids = list(dict.fromkeys(openalex_ids))
        selected_ids = unique_ids[: settings.openalex_full_text_max_documents]
        client = make_client(settings)

        async def fetch_one(value: str) -> SourceRecord | dict[str, str]:
            try:
                work_id = normalize_openalex_id(value)
                work = registry.get(work_id)
                document = await client.fetch(
                    work_id,
                    has_grobid_xml=work.has_grobid_xml if work else None,
                    has_pdf=work.has_pdf if work else None,
                )
                return SourceRecord(
                    provider="openalex_content",
                    source_type="scholarly_full_text",
                    evidence_level="full_text",
                    title=work.title if work else work_id,
                    url=(work.doi or work.landing_page_url or work.openalex_id) if work else value,
                    content=document.text,
                    published_at=(
                        work.publication_date
                        or (str(work.publication_year) if work.publication_year else None)
                    )
                    if work
                    else None,
                    authors=work.authors if work else [],
                    identifiers={
                        key: item
                        for key, item in {
                            "openalex_id": work_id,
                            "doi": work.doi if work else None,
                        }.items()
                        if item
                    },
                    metadata={
                        "content_format": document.content_format,
                        "original_character_count": document.original_character_count,
                        "returned_character_count": len(document.text),
                        "truncated": document.truncated,
                        "content_api": "https://content.openalex.org",
                    },
                )
            except Exception as exc:
                return {"openalex_id": value, "error": f"{type(exc).__name__}: {exc}"}

        try:
            outcomes = await asyncio.gather(*(fetch_one(item) for item in selected_ids))
        finally:
            await client.close()
        return json.dumps(
            {
                "provider": "openalex_content",
                "evidence_level": "full_text",
                "notice": (
                    "正文来自 OpenAlex 官方 Content API。每次成功下载会消耗 OpenAlex Content "
                    "API 额度；返回内容可能按配置截断。"
                ),
                "requested": len(openalex_ids),
                "processed": len(selected_ids),
                "results": [
                    item.model_dump(exclude_none=True)
                    if isinstance(item, SourceRecord)
                    else item
                    for item in outcomes
                ],
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        coroutine=fetch_fulltext,
        name="openalex_fetch_fulltext",
        description=(
            "根据 OpenAlex Work ID 从官方 Content API 读取论文正文。优先使用结构化 GROBID "
            "XML，必要时回退 PDF 文本解析。仅对入选论文调用；需要 OPENALEX_API_KEY，且每篇"
            "全文会消耗 OpenAlex Content API 额度。"
        ),
        args_schema=OpenAlexFullTextInput,
    )


def create_openalex_tools(settings: Settings) -> list[BaseTool]:
    """创建共享检索结果注册表的 OpenAlex 摘要与全文工具。"""

    registry: dict[str, ScholarlyWork] = {}
    tools = [create_openalex_tool(settings, registry)]
    if settings.openalex_fetch_full_text and settings.openalex_api_key:
        tools.append(create_openalex_fulltext_tool(settings, registry))
    return tools


def create_tavily_tool(
    settings: Settings,
    *,
    client: Any | None = None,
    summarization_model: BaseChatModel | Any | None = None,
    summary_cache: TavilySummaryCache | None = None,
    acquire_model_call: Callable[[str], Awaitable[bool]] | None = None,
) -> BaseTool | None:
    """创建批量搜索、URL 去重、完整结果保留和缓存复用的 Tavily 工具。"""

    if not settings.enable_tavily or (
        not settings.tavily_active_api_key and client is None
    ):
        return None
    if client is None:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(
            api_key=settings.tavily_active_api_key,
            api_base_url=settings.tavily_active_base_url,
        )
    model = summarization_model or build_chat_model(
        settings,
        max_tokens=settings.tavily_summary_max_tokens,
        disable_thinking=True,
    ).with_structured_output(WebpageSummary, method="json_schema")
    cache = summary_cache or TavilySummaryCache(settings.tavily_summary_cache_size)

    async def may_summarize() -> bool:
        if acquire_model_call is None:
            return True
        return await acquire_model_call("tavily_webpage_summary")

    async def search(queries: list[str]) -> str:
        prepared = [item.strip() for item in queries if item.strip()][
            : settings.tavily_max_queries
        ]
        if not prepared:
            return json.dumps({"provider": "tavily", "results": []}, ensure_ascii=False)

        async def execute(query: str) -> tuple[str, dict[str, Any] | Exception]:
            try:
                result = await client.search(
                    query=query,
                    max_results=settings.tavily_max_results,
                    topic=settings.tavily_topic,
                    search_depth=settings.tavily_search_depth,
                    include_answer=False,
                    include_raw_content=True,
                )
                # Tavily Hub wraps the upstream Tavily payload as
                # {code, data: {ok, data: <official response>}}.
                if "code" in result:
                    envelope = result.get("data")
                    if (
                        result.get("code") != 0
                        or not isinstance(envelope, dict)
                        or envelope.get("ok") is not True
                    ):
                        message = result.get("message") or "Tavily Hub request failed"
                        raise RuntimeError(str(message))
                    result = envelope.get("data")
                    if not isinstance(result, dict):
                        raise RuntimeError("Tavily Hub returned an invalid response payload")
                return query, result
            except Exception as exc:
                return query, exc

        responses = await asyncio.gather(*(execute(query) for query in prepared))
        unique: dict[str, dict[str, Any]] = {}
        errors: list[dict[str, str]] = []
        raw_result_count = 0
        for query, response in responses:
            if isinstance(response, Exception):
                errors.append({"query": query, "error": f"{type(response).__name__}: {response}"})
                continue
            for result in response.get("results") or []:
                raw_result_count += 1
                url = str(result.get("url") or "").strip()
                if not url:
                    continue
                canonical = _canonical_url(url)
                score = float(result.get("score") or 0)
                existing = unique.get(canonical)
                if existing is None:
                    unique[canonical] = {
                        **result,
                        "url": url,
                        "canonical_url": canonical,
                        "matched_queries": [query],
                    }
                else:
                    existing["matched_queries"] = list(
                        dict.fromkeys([*existing.get("matched_queries", []), query])
                    )
                    if score > float(existing.get("score") or 0):
                        matched_queries = existing["matched_queries"]
                        unique[canonical] = {
                            **result,
                            "url": url,
                            "canonical_url": canonical,
                            "matched_queries": matched_queries,
                        }

        candidates = sorted(
            unique.values(),
            key=lambda item: (
                float(item.get("score") or 0),
                len(item.get("matched_queries") or []),
            ),
            reverse=True,
        )
        # Do not discard lower-ranked pages. Ranking only determines processing
        # order; every unique Tavily result is retained. If the model-call budget
        # runs out, the remaining pages still return their search excerpts.
        selected = candidates

        # Select summary candidates in query rounds before filling by global
        # score. This preserves coverage across batched queries while retaining
        # the project's Top-K cost bound, so one broad query cannot crowd out the
        # remaining queries.
        summary_candidates: list[dict[str, Any]] = []
        selected_canonicals: set[str] = set()
        summary_limit = min(len(selected), settings.tavily_max_summarized_results)
        for query in prepared:
            match = next(
                (
                    item
                    for item in selected
                    if query in (item.get("matched_queries") or [])
                    and str(item["canonical_url"]) not in selected_canonicals
                ),
                None,
            )
            if match is not None and len(summary_candidates) < summary_limit:
                summary_candidates.append(match)
                selected_canonicals.add(str(match["canonical_url"]))
        for item in selected:
            canonical = str(item["canonical_url"])
            if len(summary_candidates) >= summary_limit:
                break
            if canonical not in selected_canonicals:
                summary_candidates.append(item)
                selected_canonicals.add(canonical)

        async def summarize(
            result: dict[str, Any],
            *,
            use_full_page: bool,
        ) -> tuple[SourceRecord, bool, bool]:
            url = str(result["url"])
            canonical = str(result["canonical_url"])
            raw = str(result.get("raw_content") or "").strip()
            excerpt = str(result.get("content") or "").strip()
            title = str(result.get("title") or url)
            if not raw:
                return (
                    SourceRecord(
                        provider="tavily",
                        source_type="web_page",
                        evidence_level="search_excerpt",
                        title=title,
                        url=url,
                        content=excerpt,
                        published_at=result.get("published_date"),
                        metadata={
                            "matched_queries": result.get("matched_queries"),
                            "score": result.get("score"),
                            "raw_content_available": False,
                            "summary_cache_hit": False,
                        },
                    ),
                    False,
                    False,
                )
            if not use_full_page:
                return (
                    SourceRecord(
                        provider="tavily",
                        source_type="web_page",
                        evidence_level="search_excerpt",
                        title=title,
                        url=url,
                        content=excerpt,
                        published_at=result.get("published_date"),
                        metadata={
                            "matched_queries": result.get("matched_queries"),
                            "score": result.get("score"),
                            "raw_content_available": True,
                            "summary_skipped_lower_rank": True,
                            "summary_cache_hit": False,
                        },
                    ),
                    False,
                    False,
                )
            url_lock = await cache.lock_for(canonical)
            async with url_lock:
                cached = await cache.get(canonical)
                if cached is not None:
                    cached.url = url
                    cached.metadata.update(
                        {
                            "matched_queries": result.get("matched_queries"),
                            "score": result.get("score"),
                            "summary_cache_hit": True,
                        }
                    )
                    return cached, True, False

                if not await may_summarize():
                    return (
                        SourceRecord(
                            provider="tavily",
                            source_type="web_page",
                            evidence_level="search_excerpt",
                            title=title,
                            url=url,
                            content=excerpt,
                            published_at=result.get("published_date"),
                            metadata={
                                "matched_queries": result.get("matched_queries"),
                                "score": result.get("score"),
                                "raw_content_available": True,
                                "summary_skipped_budget_exhausted": True,
                                "summary_cache_hit": False,
                            },
                        ),
                        False,
                        True,
                    )

                limited = raw[: settings.tavily_max_content_length]
                prompt = WEBPAGE_SUMMARY_PROMPT.format(
                    query="；".join(result.get("matched_queries") or []),
                    title=title,
                    url=url,
                    webpage_content=limited,
                )
                summary_failed = False
                try:
                    summary = await asyncio.wait_for(
                        model.ainvoke([HumanMessage(content=prompt)]),
                        timeout=settings.tavily_summary_timeout_seconds,
                    )
                    content = summary.summary
                    if summary.key_excerpts:
                        content += "\n\n关键原文：\n" + "\n".join(
                            f"- {item}" for item in summary.key_excerpts
                        )
                except Exception:
                    summary_failed = True
                    # Do not return tens of thousands of raw characters to the
                    # Researcher when summarization fails; the search excerpt is
                    # safer for both context size and prompt-injection exposure.
                    content = excerpt or limited[:2_000]
                record = SourceRecord(
                    provider="tavily",
                    source_type="web_page",
                    evidence_level=(
                        "search_excerpt" if summary_failed else "webpage_full_text"
                    ),
                    title=title,
                    url=url,
                    content=content,
                    published_at=result.get("published_date"),
                    metadata={
                        "matched_queries": result.get("matched_queries"),
                        "score": result.get("score"),
                        "raw_content_available": True,
                        "raw_character_count": len(raw),
                        "raw_content_truncated": len(raw) > len(limited),
                        "summary_failed_returned_excerpt": summary_failed,
                        "summary_cache_hit": False,
                    },
                )
                if not summary_failed:
                    await cache.put(canonical, record)
                return record, False, False

        full_page_summary_count = len(summary_candidates)
        full_page_canonicals = {
            str(item["canonical_url"]) for item in summary_candidates
        }
        outcomes = await asyncio.gather(
            *(
                summarize(
                    result,
                    use_full_page=str(result["canonical_url"]) in full_page_canonicals,
                )
                for result in selected
            )
        )
        records = [item[0] for item in outcomes]
        return json.dumps(
            {
                "provider": "tavily",
                "citation_rule": "以网页标题和原始 URL 引用，并区分网页原文与搜索摘要。",
                "queries": prepared,
                "errors": errors,
                "candidate_count": len(candidates),
                "duplicates_removed": max(raw_result_count - len(candidates), 0),
                "selected_for_summary": full_page_summary_count,
                "retained_count": len(records),
                "cache_hits": sum(1 for _, cache_hit, _ in outcomes if cache_hit),
                "summaries_skipped_by_budget": sum(
                    1 for _, _, skipped in outcomes if skipped
                ),
                "results": [record.model_dump(exclude_none=True) for record in records],
            },
            ensure_ascii=False,
        )

    return StructuredTool.from_function(
        coroutine=search,
        name="tavily_search",
        description=(
            "并发执行多条 Tavily 网页检索，获取 raw_content，跨查询按规范化 URL 去重；"
            "相关性排序只决定处理顺序，所有非重复网页都会保留。模型预算不足时保留搜索摘要，"
            "只有排名靠前的少量页面额外做全文总结，但不会因此丢弃其他网页。适合官网、政策、"
            "新闻、市场和通用资料。"
        ),
        args_schema=TavilySearchInput,
    )


async def load_mcp_tools(settings: Settings) -> list[BaseTool]:
    """从 MCP_SERVERS 中的全部服务器动态发现工具。"""

    if not settings.mcp_servers:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(settings.mcp_servers)
    tools = await client.get_tools()
    for loaded_tool in tools:
        loaded_tool.description = (
            f"{loaded_tool.description}\n"
            "引用要求：保留该 MCP 工具返回的原生记录标识和 URL，并在证据备忘录中注明"
            "服务与工具名称；如果没有公开 URL，不得自行编造。"
        )
    return list(tools)


class ResearchToolProvider:
    """复用本地工具状态，同时在每轮动态刷新 MCP 工具。"""

    def __init__(
        self,
        settings: Settings,
        budget_manager: Hy3CallBudgetManager | None = None,
    ) -> None:
        self.settings = settings
        self.budget_manager = budget_manager
        self._tavily_cache = TavilySummaryCache(settings.tavily_summary_cache_size)
        self._core_tools: list[BaseTool] | None = None

    def _get_core_tools(self) -> list[BaseTool]:
        if self._core_tools is None:
            tools: list[BaseTool] = [think_tool]
            if self.settings.enable_openalex:
                tools.extend(create_openalex_tools(self.settings))
            tavily = create_tavily_tool(
                self.settings,
                summary_cache=self._tavily_cache,
                acquire_model_call=(
                    self.budget_manager.acquire_current if self.budget_manager else None
                ),
            )
            if tavily is not None:
                tools.append(tavily)
            self._core_tools = tools
        return list(self._core_tools)

    async def get_tools(self) -> list[BaseTool]:
        return [*self._get_core_tools(), *(await load_mcp_tools(self.settings))]


async def get_research_tools(settings: Settings) -> list[BaseTool]:
    """为一次研究员调用创建当前配置允许使用的完整工具集。"""

    return await ResearchToolProvider(settings).get_tools()


def tool_names(tools: Sequence[BaseTool]) -> list[str]:
    """返回稳定且不重复的工具名称列表。"""

    return list(dict.fromkeys(item.name for item in tools))
