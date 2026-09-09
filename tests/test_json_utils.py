import pytest

from deep_research_assistant.json_utils import ModelOutputError, extract_json_object


def test_extract_json_object_from_fenced_response() -> None:
    result = extract_json_object('结果如下：\n```json\n{"topic": "RAG", "count": 8}\n```')

    assert result == {"topic": "RAG", "count": 8}


def test_extract_json_object_ignores_braces_inside_strings() -> None:
    result = extract_json_object('{"query": "比较 {RAG} 与 Agent", "valid": true}')

    assert result["valid"] is True


def test_extract_json_object_rejects_missing_json() -> None:
    with pytest.raises(ModelOutputError, match="没有 JSON"):
        extract_json_object("没有结构化结果")
