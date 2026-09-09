import httpx

from deep_research_assistant.config import Settings
from deep_research_assistant.openalex_client import OpenAlexClient, reconstruct_abstract


def test_reconstruct_abstract_orders_words_by_position() -> None:
    abstract = reconstruct_abstract(
        {"retrieval": [1], "augmented": [2], "generation": [3], "A": [0]}
    )

    assert abstract == "A retrieval augmented generation"


def test_openalex_client_normalizes_work() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "test-key"
        assert "from_publication_date:2023-01-01" in request.url.params["filter"]
        assert request.url.params["sort"] == "relevance_score:desc"
        return httpx.Response(
            200,
            json={
                "meta": {"count": 1},
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "doi": "https://doi.org/10.1/example",
                        "display_name": "Evaluating Retrieval-Augmented Generation",
                        "publication_year": 2024,
                        "publication_date": "2024-04-01",
                        "type": "article",
                        "authorships": [{"author": {"display_name": "Ada Researcher"}}],
                        "primary_location": {
                            "landing_page_url": "https://example.org/paper",
                            "pdf_url": "https://example.org/paper.pdf",
                            "source": {"display_name": "Example Journal"},
                        },
                        "abstract_inverted_index": {"RAG": [0], "evaluation": [1]},
                        "has_content": {"pdf": True, "grobid_xml": True},
                        "cited_by_count": 42,
                        "open_access": {"is_oa": True, "oa_url": "https://example.org/paper.pdf"},
                        "topics": [{"display_name": "Information Retrieval"}],
                    }
                ],
            },
        )

    settings = Settings(_env_file=None, openalex_api_key="test-key")
    http_client = httpx.Client(
        base_url="https://api.openalex.org",
        transport=httpx.MockTransport(handler),
    )
    client = OpenAlexClient(settings=settings, client=http_client)

    works = client.search(
        "retrieval augmented generation evaluation",
        query_id="Q1",
        start_year=2023,
        end_year=2026,
    )

    assert len(works) == 1
    assert works[0].authors == ["Ada Researcher"]
    assert works[0].abstract == "RAG evaluation"
    assert works[0].has_full_text is True
    assert works[0].has_pdf is True
    assert works[0].has_grobid_xml is True
    assert works[0].matched_query_ids == ["Q1"]
