"""Generate bounded OpenAlex queries from a confirmed research outline."""

from deep_research_assistant.hy3_client import Hy3Client
from deep_research_assistant.models import LiteratureSearchPlan, ResearchIntent, ResearchOutline

SYSTEM_PROMPT = """你是学术文献检索规划节点。根据调研意图和大纲生成 OpenAlex 检索词。
规则：
1. 每个需要外部证据的一级章节至少对应一个 query，整份计划最多 8 个 query。
2. query 必须是简洁的英文主题检索词，适用于论文标题和摘要全文搜索。
3. 每个 query 聚焦一个概念组合，优先使用领域通行术语和完整英文名称。
4. 不使用 site:、字段前缀或数据库专用过滤语法；年份由程序单独过滤。
5. query_id 从 Q1 开始连续编号，section_id 必须来自输入大纲。
6. research_question 原样对应大纲中的一个问题；rationale 简述该检索式覆盖什么。
7. 避免只有一个宽泛词，也避免把整句话翻译成超长查询。
8. 不为“总结”“结论”“最佳实践”等无法直接检索的章节生成宽泛查询；复用前文证据。
9. 输出必须符合指定 JSON Schema。
"""


class SearchPlanner:
    """Use Hy3 to transform outline questions into executable search queries."""

    def __init__(self, client: Hy3Client) -> None:
        self.client = client

    def generate(
        self,
        intent: ResearchIntent,
        outline: ResearchOutline,
        feedback: str | None = None,
    ) -> LiteratureSearchPlan:
        payload = "调研意图：\n" + intent.model_dump_json(indent=2)
        payload += "\n\n调研大纲：\n" + outline.model_dump_json(indent=2)
        if feedback:
            payload += "\n\n上一轮检索质量反馈，请据此改写检索式：\n" + feedback.strip()
        plan = self.client.chat_structured(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            LiteratureSearchPlan,
            schema_name="literature_search_plan",
            stage="search",
        )
        return plan.model_copy(update={"queries": plan.queries[:8]})
