"""Utilities for validating structured model output."""

import json
from typing import Any


class ModelOutputError(ValueError):
    """Raised when a model response does not contain a valid JSON object."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from plain text or a fenced block."""

    start = text.find("{")
    if start < 0:
        raise ModelOutputError("模型响应中没有 JSON 对象")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ModelOutputError(f"模型返回了无效 JSON：{exc.msg}") from exc
                if not isinstance(value, dict):
                    raise ModelOutputError("模型响应的顶层 JSON 必须是对象")
                return value

    raise ModelOutputError("模型响应中的 JSON 对象不完整")
