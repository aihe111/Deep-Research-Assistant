"""Deterministically preserve explicit constraints from the original request."""

import re

from deep_research_assistant.models import ReportDepth, ReportLanguage, ResearchIntent

YEAR_RANGE_PATTERN = re.compile(
    r"(?P<start>(?:19|20)\d{2})\s*(?:-|–|—|~|～|至|到)\s*"
    r"(?P<end>(?:19|20)\d{2})\s*年?"
)
AUDIENCE_PATTERN = re.compile(r"面向(?P<audience>[^，。；,;\n]{2,40})")
SOURCE_COUNT_PATTERN = re.compile(r"(?P<count>\d{1,2})\s*篇(?:核心)?(?:论文|文献|资料)")
WORD_COUNT_PATTERN = re.compile(r"(?P<count>\d{3,5})\s*(?:字|字符|words?)", re.IGNORECASE)
FOCUS_PATTERN = re.compile(
    r"(?:重点关注|重点研究|关注|聚焦于?|侧重于?)(?P<focus>[^，。；;\n]{2,120})"
)

WORD_COUNT_QUESTION = "期望最终报告大约多少字？请输入 500-20000 之间的数字，例如“3000字”。"
AUDIENCE_QUESTION = "报告的目标读者是谁？例如本科生、研究生或有基础的大模型开发者。"
TIME_RANGE_QUESTION = "希望覆盖什么时间范围？例如“2023-2026年”或“不限时间”。"
FOCUS_QUESTION = "报告重点关注哪些方向？请列出 1-3 个具体方面。"


def normalize_explicit_constraints(request: str, intent: ResearchIntent) -> ResearchIntent:
    """Override model guesses only when a constraint is explicit in the request."""

    updates: dict[str, object] = {}

    year_range = YEAR_RANGE_PATTERN.search(request)
    if year_range:
        updates["start_year"] = int(year_range.group("start"))
        updates["end_year"] = int(year_range.group("end"))

    audience_match = AUDIENCE_PATTERN.search(request)
    if audience_match:
        updates["audience"] = audience_match.group("audience").strip()

    source_count_match = SOURCE_COUNT_PATTERN.search(request)
    if source_count_match:
        updates["target_source_count"] = int(source_count_match.group("count"))

    word_count_match = WORD_COUNT_PATTERN.search(request)
    if word_count_match:
        word_count = int(word_count_match.group("count"))
        if 500 <= word_count <= 20000:
            updates["target_word_count"] = word_count

    focus_match = FOCUS_PATTERN.search(request)
    if focus_match:
        focus_areas = [
            item.strip().removesuffix("等")
            for item in re.split(r"、|以及|及|与|和", focus_match.group("focus"))
            if item.strip()
        ]
        if focus_areas:
            updates["focus_areas"] = focus_areas[:5]

    if "中文" in request:
        updates["report_language"] = ReportLanguage.CHINESE
    elif "英文" in request:
        updates["report_language"] = ReportLanguage.ENGLISH

    if "标准深度" in request:
        updates["depth"] = ReportDepth.STANDARD
    elif any(marker in request for marker in ("深入报告", "深度报告", "详细报告")):
        updates["depth"] = ReportDepth.DEEP
    elif any(marker in request for marker in ("简短报告", "简要报告", "概览")):
        updates["depth"] = ReportDepth.BRIEF

    normalized = ResearchIntent.model_validate({**intent.model_dump(), **updates})
    questions = _required_clarification_questions(request, normalized)
    return ResearchIntent.model_validate(
        {**normalized.model_dump(), "clarification_questions": questions}
    )


def _required_clarification_questions(request: str, intent: ResearchIntent) -> list[str]:
    """Combine model questions with deterministic checks for essential parameters."""

    blocking_keywords = (
        "主题",
        "目标",
        "用途",
        "受众",
        "读者",
        "时间",
        "年份",
        "重点",
        "关注",
        "字数",
        "篇幅",
    )
    questions = [
        question
        for question in intent.clarification_questions
        if any(keyword in question for keyword in blocking_keywords)
    ]
    if intent.target_word_count is None:
        questions.append(WORD_COUNT_QUESTION)
    else:
        questions = [
            question for question in questions if "字数" not in question and "篇幅" not in question
        ]

    audience_is_explicit = any(marker in request for marker in ("面向", "目标读者", "受众"))
    if not audience_is_explicit:
        questions.append(AUDIENCE_QUESTION)
    else:
        questions = [
            question
            for question in questions
            if "受众" not in question and "目标读者" not in question
        ]

    time_is_explicit = bool(YEAR_RANGE_PATTERN.search(request)) or any(
        marker in request for marker in ("不限时间", "近几年", "经典文献", "时间范围")
    )
    if not time_is_explicit:
        questions.append(TIME_RANGE_QUESTION)
    else:
        questions = [
            question
            for question in questions
            if "时间范围" not in question and "年份" not in question
        ]

    focus_is_explicit = any(marker in request for marker in ("重点", "关注", "聚焦", "侧重"))
    if not focus_is_explicit:
        questions.append(FOCUS_QUESTION)
    else:
        questions = [
            question
            for question in questions
            if "重点" not in question and "关注方向" not in question
        ]

    return list(dict.fromkeys(question.strip() for question in questions if question.strip()))[:4]
