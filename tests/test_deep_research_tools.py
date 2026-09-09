import asyncio
import json

from deep_research_assistant.config import Settings
from deep_research_assistant.deep_research_state import WebpageSummary
from deep_research_assistant.deep_research_tools import (
    create_openalex_fulltext_tool,
    create_openalex_tool,
    create_tavily_tool,
)
from deep_research_assistant.models import ScholarlyWork
from deep_research_assistant.scholarly_fulltext import FullTextDocument


def test_openalex_is_exposed_as_provider_aware_tool(monkeypatch) -> None:
    class FakeOpenAlex:
        def __init__(self, settings) -> None:
            pass

        def close(self) -> None:
            pass

        def search(self, *args, **kwargs):
            return [
                ScholarlyWork(
                    openalex_id="https://openalex.org/W1",
                    doi="https://doi.org/10.1/example",
                    title="Evidence Paper",
                    authors=["A. Author"],
                    publication_year=2025,
                    abstract="Grounded finding.",
                    cited_by_count=7,
                )
            ]

    monkeypatch.setattr("deep_research_assistant.deep_research_tools.OpenAlexClient", FakeOpenAlex)
    tool = create_openalex_tool(Settings(_env_file=None))
    payload = json.loads(asyncio.run(tool.ainvoke({"query": "evidence", "max_results": 3})))

    assert tool.name == "openalex_search"
    assert payload["provider"] == "openalex"
    assert payload["results"][0]["identifiers"]["openalex_id"].endswith("W1")
    assert payload["results"][0]["url"] == "https://doi.org/10.1/example"


def test_tavily_is_optional_without_api_key() -> None:
    assert create_tavily_tool(Settings(_env_file=None, tavily_api_key="")) is None


def test_tavily_hub_configures_gateway_client(monkeypatch) -> None:
    captured = {}

    class FakeTavilyClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("tavily.AsyncTavilyClient", FakeTavilyClient)

    tool = create_tavily_tool(
        Settings(
            _env_file=None,
            tavily_api_key="tvly-official",
            tavily_hub_api_key="thb-hub",
        ),
        summarization_model=object(),
    )

    assert tool is not None
    assert captured == {
        "api_key": "thb-hub",
        "api_base_url": "https://tavily.sharyuke.com/api/proxy",
    }


def test_tavily_hub_response_is_unwrapped() -> None:
    class FakeTavilyHubClient:
        async def search(self, *, query, **kwargs):
            return {
                "code": 0,
                "message": "ok",
                "data": {
                    "ok": True,
                    "data": {
                        "query": query,
                        "results": [
                            {
                                "title": "Hub result",
                                "url": "https://example.com/hub",
                                "content": "Hub excerpt",
                                "score": 0.9,
                            }
                        ],
                    },
                    "credits": 1,
                },
            }

    tool = create_tavily_tool(
        Settings(_env_file=None, tavily_hub_api_key="thb-hub"),
        client=FakeTavilyHubClient(),
        summarization_model=object(),
    )
    assert tool is not None

    payload = json.loads(asyncio.run(tool.ainvoke({"queries": ["gateway"]})))

    assert payload["errors"] == []
    assert payload["retained_count"] == 1
    assert payload["results"][0]["title"] == "Hub result"


def test_openalex_fulltext_tool_returns_full_text_evidence() -> None:
    work = ScholarlyWork(
        openalex_id="https://openalex.org/W1",
        doi="https://doi.org/10.1/example",
        title="Full Paper",
        authors=["A. Author"],
        has_full_text=True,
        has_grobid_xml=True,
    )

    class FakeFullTextClient:
        def __init__(self, settings) -> None:
            pass

        async def fetch(self, openalex_id, **kwargs):
            return FullTextDocument(
                openalex_id=openalex_id,
                content_format="grobid_xml",
                text="Methods and complete results.",
                original_character_count=29,
                truncated=False,
            )

        async def close(self) -> None:
            pass

    settings = Settings(_env_file=None, openalex_api_key="test-key")
    tool = create_openalex_fulltext_tool(
        settings,
        {"W1": work},
        client_factory=FakeFullTextClient,
    )
    payload = json.loads(asyncio.run(tool.ainvoke({"openalex_ids": ["W1"]})))

    assert tool.name == "openalex_fetch_fulltext"
    assert payload["evidence_level"] == "full_text"
    assert payload["results"][0]["title"] == "Full Paper"
    assert payload["results"][0]["content"] == "Methods and complete results."


def test_tavily_batches_queries_deduplicates_urls_and_summarizes_raw_content() -> None:
    shared = {"active": 0, "max_active": 0}

    class FakeTavilyClient:
        async def search(self, *, query, **kwargs):
            assert kwargs["include_raw_content"] is True
            shared["active"] += 1
            shared["max_active"] = max(shared["max_active"], shared["active"])
            await asyncio.sleep(0.01)
            shared["active"] -= 1
            return {
                "results": [
                    {
                        "title": "Primary page",
                        "url": "https://example.com/shared",
                        "content": "short excerpt",
                        "raw_content": f"complete page for {query}",
                        "score": 0.9,
                    }
                ]
            }

    class FakeSummaryModel:
        async def ainvoke(self, messages):
            return WebpageSummary(summary="正文摘要", key_excerpts=["关键原文"])

    tool = create_tavily_tool(
        Settings(_env_file=None, tavily_api_key="test-key"),
        client=FakeTavilyClient(),
        summarization_model=FakeSummaryModel(),
    )
    assert tool is not None
    payload = json.loads(
        asyncio.run(tool.ainvoke({"queries": ["first query", "second query"]}))
    )

    assert shared["max_active"] == 2
    assert payload["queries"] == ["first query", "second query"]
    assert len(payload["results"]) == 1
    assert payload["duplicates_removed"] == 1
    assert payload["retained_count"] == 1
    assert payload["results"][0]["evidence_level"] == "webpage_full_text"
    assert "关键原文" in payload["results"][0]["content"]


def test_tavily_retains_all_unique_results_and_only_summarizes_top_k() -> None:
    shared = {"summary_calls": 0}

    class FakeTavilyClient:
        async def search(self, *, query, **kwargs):
            return {
                "results": [
                    {
                        "title": f"Page {index}",
                        "url": f"https://example.com/{index}?utm_source={query}",
                        "content": f"excerpt {index}",
                        "raw_content": f"raw page {index}",
                        "score": index / 10,
                    }
                    for index in range(1, 7)
                ]
            }

    class FakeSummaryModel:
        async def ainvoke(self, messages):
            shared["summary_calls"] += 1
            return WebpageSummary(summary="缓存摘要", key_excerpts=[])

    tool = create_tavily_tool(
        Settings(
            _env_file=None,
            tavily_api_key="test-key",
            tavily_max_summarized_results=3,
        ),
        client=FakeTavilyClient(),
        summarization_model=FakeSummaryModel(),
    )
    assert tool is not None

    first = json.loads(asyncio.run(tool.ainvoke({"queries": ["first", "second"]})))
    second = json.loads(asyncio.run(tool.ainvoke({"queries": ["third"]})))

    assert first["candidate_count"] == 6
    assert first["selected_for_summary"] == 3
    assert first["retained_count"] == 6
    assert first["duplicates_removed"] == 6
    assert [item["title"] for item in first["results"]] == [
        "Page 6",
        "Page 5",
        "Page 4",
        "Page 3",
        "Page 2",
        "Page 1",
    ]
    assert [item["evidence_level"] for item in first["results"]] == [
        "webpage_full_text",
        "webpage_full_text",
        "webpage_full_text",
        "search_excerpt",
        "search_excerpt",
        "search_excerpt",
    ]
    assert shared["summary_calls"] == 3
    assert second["cache_hits"] == 3
    assert second["retained_count"] == 6
    assert all(
        item["metadata"]["summary_cache_hit"] for item in second["results"][:3]
    )
    assert all(
        item["metadata"]["summary_skipped_lower_rank"]
        for item in second["results"][3:]
    )


def test_tavily_stops_summaries_when_global_model_budget_is_exhausted() -> None:
    shared = {"budget_checks": 0, "summary_calls": 0}

    class FakeTavilyClient:
        async def search(self, *, query, **kwargs):
            return {
                "results": [
                    {
                        "title": f"Page {index}",
                        "url": f"https://example.com/budget-{index}",
                        "content": f"excerpt {index}",
                        "raw_content": f"raw page {index}",
                        "score": 1 - index / 10,
                    }
                    for index in range(3)
                ]
            }

    class FakeSummaryModel:
        async def ainvoke(self, messages):
            shared["summary_calls"] += 1
            return WebpageSummary(summary="预算内摘要", key_excerpts=[])

    async def acquire(stage: str) -> bool:
        assert stage == "tavily_webpage_summary"
        shared["budget_checks"] += 1
        return shared["budget_checks"] <= 1

    tool = create_tavily_tool(
        Settings(
            _env_file=None,
            tavily_api_key="test-key",
            tavily_max_summarized_results=3,
        ),
        client=FakeTavilyClient(),
        summarization_model=FakeSummaryModel(),
        acquire_model_call=acquire,
    )
    assert tool is not None
    payload = json.loads(asyncio.run(tool.ainvoke({"queries": ["budget"]})))

    assert shared["summary_calls"] == 1
    assert payload["summaries_skipped_by_budget"] == 2
    assert sum(
        item["metadata"].get("summary_skipped_budget_exhausted", False)
        for item in payload["results"]
    ) == 2


def test_tavily_summary_top_k_covers_each_batched_query_before_score_fill() -> None:
    class FakeTavilyClient:
        async def search(self, *, query, **kwargs):
            if query == "pricing":
                return {
                    "results": [
                        {
                            "title": "Pricing one",
                            "url": "https://example.com/pricing-1",
                            "content": "pricing excerpt one",
                            "raw_content": "pricing full page one",
                            "score": 0.99,
                        },
                        {
                            "title": "Pricing two",
                            "url": "https://example.com/pricing-2",
                            "content": "pricing excerpt two",
                            "raw_content": "pricing full page two",
                            "score": 0.98,
                        },
                    ]
                }
            return {
                "results": [
                    {
                        "title": "Context limit",
                        "url": "https://example.com/context",
                        "content": "context excerpt",
                        "raw_content": "context full page",
                        "score": 0.20,
                    }
                ]
            }

    class FakeSummaryModel:
        async def ainvoke(self, messages):
            return WebpageSummary(summary="完整页面摘要", key_excerpts=[])

    tool = create_tavily_tool(
        Settings(
            _env_file=None,
            tavily_api_key="test-key",
            tavily_max_summarized_results=2,
        ),
        client=FakeTavilyClient(),
        summarization_model=FakeSummaryModel(),
    )
    assert tool is not None
    payload = json.loads(
        asyncio.run(tool.ainvoke({"queries": ["pricing", "context window"]}))
    )

    evidence_by_title = {
        item["title"]: item["evidence_level"] for item in payload["results"]
    }
    assert evidence_by_title["Pricing one"] == "webpage_full_text"
    assert evidence_by_title["Context limit"] == "webpage_full_text"
    assert evidence_by_title["Pricing two"] == "search_excerpt"
