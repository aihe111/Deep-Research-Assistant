"""Turn a natural-language request into a structured research intent."""

from deep_research_assistant.hy3_client import Hy3Client
from deep_research_assistant.intent_normalizer import normalize_explicit_constraints
from deep_research_assistant.models import ResearchIntent

SYSTEM_PROMPT = """你是 AI 与计算机领域调研助手中的需求分析节点。
你的任务是把用户的自然语言请求整理为结构化调研意图，不执行检索，也不撰写报告。
遵守以下规则：
1. 逐项识别用户明确表达的主题、目标、受众、关注点、排除项、年份、语言、深度、
   报告字数、来源类型和数量；不要漏掉数字与时间范围。类似“2023-2026 年”必须分别写入
   start_year=2023 和 end_year=2026。
2. focus_areas 应是可检索的具体方向，避免空泛词语。
3. 规范化枚举值：中文报告使用 zh-CN，英文报告使用 en；标准深度使用 standard。
4. 用户未指定文献数量时，target_source_count 必须使用 8；不得自行扩大数量。
5. source_types 只能从 paper、survey、dataset、official_doc 中选择。
6. target_word_count 表示最终报告目标字数；用户未指定时必须为 null，不能自行猜测。
7. 如果缺失信息会明显改变调研范围，在 clarification_questions 中提出，最多 4 个。
8. 不要虚构用户身份、截止时间或指定文献。
9. 输出必须符合指定 JSON Schema。
"""


class IntentAnalyzer:
    """Analyze a user's research request with Hy3."""

    def __init__(self, client: Hy3Client) -> None:
        self.client = client

    def analyze(self, request: str) -> ResearchIntent:
        if len(request.strip()) < 4:
            raise ValueError("调研需求过短，请至少描述主题和目标")

        intent = self.client.chat_structured(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.strip()},
            ],
            ResearchIntent,
            schema_name="research_intent",
            stage="intent",
        )
        return normalize_explicit_constraints(request, intent)
