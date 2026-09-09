"""通过 OpenAlex 官方 Content API 获取并解析开放学术全文。"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Literal

import httpx

from deep_research_assistant.config import Settings


class ScholarlyFullTextError(RuntimeError):
    """OpenAlex 全文无法安全下载或解析。"""


@dataclass(frozen=True)
class FullTextDocument:
    openalex_id: str
    content_format: Literal["grobid_xml", "pdf"]
    text: str
    original_character_count: int
    truncated: bool


def normalize_openalex_id(value: str) -> str:
    """把 URL 或短 ID 规范化为 Content API 接受的 W… 标识。"""

    match = re.search(r"(?:^|/)(W\d+)$", value.strip(), flags=re.IGNORECASE)
    if not match:
        raise ScholarlyFullTextError(f"无效的 OpenAlex Work ID：{value}")
    return match.group(1).upper()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def extract_grobid_text(payload: bytes) -> str:
    """从 OpenAlex 的 TEI XML 中提取标题层级和正文段落。"""

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ScholarlyFullTextError("OpenAlex 返回的 GROBID XML 无法解析") from exc
    body = next((item for item in root.iter() if _local_name(item.tag) == "body"), None)
    if body is None:
        raise ScholarlyFullTextError("GROBID XML 中没有论文正文")
    blocks: list[str] = []
    for item in body.iter():
        if _local_name(item.tag) not in {"head", "p", "item"}:
            continue
        text = " ".join("".join(item.itertext()).split())
        if text and (not blocks or text != blocks[-1]):
            blocks.append(text)
    content = "\n\n".join(blocks).strip()
    if not content:
        raise ScholarlyFullTextError("GROBID XML 正文为空")
    return content


def extract_pdf_text(payload: bytes) -> str:
    """使用 PyMuPDF 从 PDF 字节中逐页提取正文。"""

    import fitz

    try:
        document = fitz.open(stream=payload, filetype="pdf")
        pages = [page.get_text("text", sort=True).strip() for page in document]
        document.close()
    except Exception as exc:
        raise ScholarlyFullTextError("OpenAlex PDF 无法解析") from exc
    content = "\n\n".join(page for page in pages if page).strip()
    if not content:
        raise ScholarlyFullTextError("OpenAlex PDF 没有可提取文本")
    return content


class OpenAlexFullTextClient:
    """限量调用 OpenAlex Content API，优先读取结构化 GROBID XML。"""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        if not settings.openalex_api_key:
            raise ScholarlyFullTextError("读取 OpenAlex 全文需要配置 OPENALEX_API_KEY")
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url="https://content.openalex.org",
            timeout=settings.openalex_full_text_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "Deep-Research-Assistant/0.3"},
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _download(self, work_id: str, extension: str) -> bytes | None:
        try:
            async with self._client.stream(
                "GET",
                f"/works/{work_id}.{extension}",
                params={"api_key": self.settings.openalex_api_key},
            ) as response:
                if response.status_code == 404:
                    return None
                if response.status_code in {401, 403}:
                    raise ScholarlyFullTextError(
                        "OpenAlex Content API 拒绝访问，请检查 API Key 和额度"
                    )
                response.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.settings.openalex_full_text_max_bytes:
                        raise ScholarlyFullTextError(
                            "论文文件超过 OPENALEX_FULL_TEXT_MAX_BYTES 限制"
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
        except httpx.HTTPError as exc:
            raise ScholarlyFullTextError(f"OpenAlex 全文下载失败：{exc}") from exc

    async def fetch(
        self,
        openalex_id: str,
        *,
        has_grobid_xml: bool | None = None,
        has_pdf: bool | None = None,
    ) -> FullTextDocument:
        work_id = normalize_openalex_id(openalex_id)
        attempts: list[tuple[str, Literal["grobid_xml", "pdf"]]] = []
        if has_grobid_xml is not False:
            attempts.append(("grobid-xml", "grobid_xml"))
        if has_pdf is not False:
            attempts.append(("pdf", "pdf"))
        if not attempts:
            raise ScholarlyFullTextError(f"{work_id} 没有可下载的全文格式")

        errors: list[str] = []
        for extension, content_format in attempts:
            payload = await self._download(work_id, extension)
            if payload is None:
                continue
            try:
                if content_format == "grobid_xml":
                    text = extract_grobid_text(payload)
                else:
                    text = await asyncio.to_thread(extract_pdf_text, payload)
            except ScholarlyFullTextError as exc:
                errors.append(str(exc))
                continue
            original_count = len(text)
            limit = self.settings.openalex_full_text_max_characters
            return FullTextDocument(
                openalex_id=work_id,
                content_format=content_format,
                text=text[:limit],
                original_character_count=original_count,
                truncated=original_count > limit,
            )
        detail = "；".join(errors) if errors else "Content API 没有返回可用文件"
        raise ScholarlyFullTextError(f"{work_id} 无法取得全文：{detail}")
