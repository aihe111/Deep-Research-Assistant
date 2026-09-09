"""Generate an editable research outline from a validated intent."""

from deep_research_assistant.hy3_client import Hy3Client
from deep_research_assistant.models import ResearchIntent, ResearchOutline

SYSTEM_PROMPT = """你是 AI 与计算机领域调研助手中的大纲规划节点。
根据已经结构化的调研意图生成一份可执行、可检索的调研大纲。
遵守以下规则：
1. 大纲必须直接服务于用户目标和关注点，不要添加无关的通用章节。
2. 正文保持 4-8 个一级章节；每章要有清晰目标和 1-4 个可检索问题。
3. 应包含问题背景、核心技术或观点比较、证据分析、局限和结论，但标题可按主题调整。
4. 避免不同章节重复回答同一个问题。
5. section_id 从 S1 开始连续编号。
6. 输出必须符合指定 JSON Schema。
"""


class OutlineGenerator:
    """Generate a research outline with Hy3."""

    def __init__(self, client: Hy3Client) -> None:
        self.client = client

    def generate(self, intent: ResearchIntent, feedback: str | None = None) -> ResearchOutline:
        user_content = "请根据以下调研意图生成大纲：\n" + intent.model_dump_json(indent=2)
        if feedback:
            user_content += "\n\n用户对上一版大纲的修改意见：\n" + feedback.strip()
        return self.client.chat_structured(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            ResearchOutline,
            schema_name="research_outline",
            stage="outline",
        )
