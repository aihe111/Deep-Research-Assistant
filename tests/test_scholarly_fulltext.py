import asyncio

import httpx
import pytest

from deep_research_assistant.config import Settings
from deep_research_assistant.scholarly_fulltext import (
    OpenAlexFullTextClient,
    ScholarlyFullTextError,
    extract_grobid_text,
    extract_pdf_text,
    normalize_openalex_id,
)


def test_extract_grobid_text_reads_only_body_blocks() -> None:
    payload = b"""<?xml version='1.0' encoding='UTF-8'?>
    <TEI xmlns='http://www.tei-c.org/ns/1.0'>
      <text><front><p>Abstract only</p></front><body>
        <div><head>Methods</head><p>We evaluated the complete dataset.</p></div>
        <div><head>Results</head><p>Accuracy improved.</p></div>
      </body></text>
    </TEI>"""

    text = extract_grobid_text(payload)

    assert "Methods" in text
    assert "Accuracy improved." in text
    assert "Abstract only" not in text


def test_normalize_openalex_id_rejects_non_work_ids() -> None:
    assert normalize_openalex_id("https://openalex.org/W3038568908") == "W3038568908"
    with pytest.raises(ScholarlyFullTextError):
        normalize_openalex_id("https://example.com/not-a-work")


def test_extract_pdf_text_reads_generated_pdf() -> None:
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Complete methods and results")
    payload = document.tobytes()
    document.close()

    assert "Complete methods and results" in extract_pdf_text(payload)


def test_content_client_uses_official_openalex_endpoint() -> None:
    xml = b"""<TEI xmlns='http://www.tei-c.org/ns/1.0'><text><body>
    <div><head>Results</head><p>Full text evidence.</p></div>
    </body></text></TEI>"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/works/W123.grobid-xml"
        assert request.url.params["api_key"] == "test-key"
        return httpx.Response(200, content=xml)

    async def run():
        http_client = httpx.AsyncClient(
            base_url="https://content.openalex.org",
            transport=httpx.MockTransport(handler),
        )
        client = OpenAlexFullTextClient(
            Settings(_env_file=None, openalex_api_key="test-key"),
            client=http_client,
        )
        try:
            return await client.fetch("W123", has_grobid_xml=True, has_pdf=False)
        finally:
            await http_client.aclose()

    result = asyncio.run(run())

    assert result.content_format == "grobid_xml"
    assert "Full text evidence." in result.text
