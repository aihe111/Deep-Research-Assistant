"""Application configuration loaded from environment variables or a local .env file."""

from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the general-purpose deep research agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hy3_base_url: str = Field(default="https://tokenhub.tencentmaas.com/v1")
    hy3_api_key: str = Field(default="")
    hy3_model: str = Field(default="hy3")
    hy3_timeout_seconds: float = Field(default=120.0, gt=0)
    hy3_reasoning_effort: str = Field(
        default="low",
        pattern=r"^(no_think|low|medium|high)$",
    )
    hy3_max_retries: int = Field(default=2, ge=0, le=5)
    hy3_retry_base_seconds: float = Field(default=0.5, ge=0, le=10)

    openalex_base_url: str = Field(default="https://api.openalex.org")
    openalex_api_key: str = Field(default="")
    openalex_email: str = Field(default="")
    openalex_per_query: int = Field(default=30, ge=5, le=100)

    # Research tools. OpenAlex is useful for scholarly evidence, while Tavily
    # and dynamically loaded MCP tools make the same graph usable in any domain.
    enable_openalex: bool = True
    enable_tavily: bool = True
    tavily_api_key: str = Field(default="")
    tavily_base_url: str = Field(default="https://api.tavily.com")
    # Tavily Hub is an optional third-party gateway. When its key is set it
    # takes precedence, while the official key remains available as a fallback.
    tavily_hub_api_key: str = Field(default="")
    tavily_hub_base_url: str = Field(
        default="https://tavily.sharyuke.com/api/proxy"
    )
    tavily_max_results: int = Field(default=20, ge=1, le=20)
    tavily_max_queries: int = Field(default=4, ge=1, le=10)
    tavily_topic: str = Field(default="general", pattern=r"^(general|news|finance)$")
    tavily_search_depth: str = Field(default="basic", pattern=r"^(basic|advanced)$")
    tavily_max_content_length: int = Field(default=50_000, ge=1_000, le=200_000)
    # This limits costly full-page LLM summaries, not retained search results.
    tavily_max_summarized_results: int = Field(default=5, ge=1, le=20)
    tavily_summary_cache_size: int = Field(default=512, ge=16, le=10_000)
    tavily_summary_max_tokens: int = Field(default=1_200, ge=256, le=16_000)
    tavily_summary_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    openalex_fetch_full_text: bool = True
    openalex_full_text_max_documents: int = Field(default=3, ge=1, le=10)
    openalex_full_text_max_characters: int = Field(default=50_000, ge=5_000, le=200_000)
    openalex_full_text_max_bytes: int = Field(
        default=20 * 1024 * 1024,
        ge=1 * 1024 * 1024,
        le=100 * 1024 * 1024,
    )
    openalex_full_text_timeout_seconds: float = Field(default=45.0, gt=0, le=300)
    mcp_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    mcp_prompt: str = Field(
        default=(
            "当 MCP 工具包含公开网页和学术检索无法提供的第一方、私有或专业领域信息时，"
            "优先使用相应 MCP 工具。"
        )
    )

    # Agent topology and bounded loops.
    allow_clarification: bool = True
    require_outline_confirmation: bool = True
    max_outline_revisions: int = Field(default=3, ge=0, le=10)
    max_concurrent_research_units: int = Field(default=5, ge=1, le=20)
    max_supervisor_iterations: int = Field(default=6, ge=1, le=20)
    # Five searches interleaved with required think-tool reflections need up to
    # ten ReAct turns.
    max_researcher_iterations: int = Field(default=10, ge=1, le=30)
    max_search_tool_calls: int = Field(default=5, ge=1, le=10)
    # Zero disables the run-wide hard cap. Per-researcher, supervisor, search,
    # and retry iteration limits still bound the graph.
    max_hy3_calls_per_research: int = Field(default=0, ge=0, le=500)
    researcher_tool_result_max_characters: int = Field(default=20_000, ge=2_000, le=100_000)
    researcher_context_max_characters: int = Field(default=60_000, ge=5_000, le=300_000)
    compression_input_max_characters: int = Field(default=50_000, ge=5_000, le=100_000)
    research_memo_target_characters: int = Field(default=7_000, ge=1_000, le=20_000)
    research_memo_max_characters: int = Field(default=10_000, ge=1_000, le=30_000)
    final_report_context_max_characters: int = Field(default=120_000, ge=10_000, le=500_000)
    research_brief_max_tokens: int = Field(default=1600, ge=256, le=16000)
    # The supervisor may spend part of the completion budget on hidden
    # reasoning before it emits tool calls. Keep enough room for both.
    supervisor_max_tokens: int = Field(default=6000, ge=256, le=16000)
    researcher_max_tokens: int = Field(default=3000, ge=256, le=32000)
    # Compression keeps comprehensive findings and runs without hidden
    # reasoning, so the output budget is available to the evidence text.
    compression_max_tokens: int = Field(default=12000, ge=512, le=32000)
    # The final writer keeps low reasoning on the first attempt and falls back
    # to a no-thinking retry when the provider returns an incomplete response.
    final_report_max_tokens: int = Field(default=16000, ge=1000, le=64000)

    # LangSmith tracing is picked up automatically by LangChain. Keeping the
    # values in Settings also makes startup diagnostics and evaluation explicit.
    langsmith_tracing: bool = False
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="deep-research-assistant")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")

    # Offline LLM-as-judge evaluation. This uses DeepSeek's OpenAI-compatible
    # Chat Completions API and writes every dimension back as LangSmith feedback.
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    evaluation_judge_model: str = Field(default="deepseek-v4-flash")
    evaluation_judge_max_tokens: int = Field(default=1500, ge=256, le=8000)
    evaluation_judge_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    evaluation_judge_max_retries: int = Field(default=2, ge=0, le=5)
    database_url: str = Field(default="sqlite:///data/deep_research_assistant.db")

    # The Web application submits durable Thread/Run work to LangGraph Server.
    langgraph_api_url: str = Field(default="http://127.0.0.1:2024")
    langgraph_api_key: str = Field(default="")
    langgraph_assistant_id: str = Field(default="deep_research_assistant")
    langgraph_request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    @property
    def tavily_active_api_key(self) -> str:
        """Return the gateway key when configured, otherwise the official key."""

        return self.tavily_hub_api_key or self.tavily_api_key

    @property
    def tavily_active_base_url(self) -> str:
        """Return the endpoint paired with the currently selected Tavily key."""

        if self.tavily_hub_api_key:
            return self.tavily_hub_base_url
        return self.tavily_base_url

    @property
    def tavily_provider(self) -> str:
        """Describe the selected Tavily provider for startup diagnostics."""

        return "hub" if self.tavily_hub_api_key else "official"

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def parse_mcp_servers(cls, value: Any) -> Any:
        """Accept either a JSON object from env or an already parsed mapping."""

        if value in (None, ""):
            return {}
        return value

    @model_validator(mode="after")
    def validate_research_memo_limits(self) -> "Settings":
        """Keep the soft memo target inside the enforced delivery limit."""

        if self.research_memo_target_characters > self.research_memo_max_characters:
            raise ValueError(
                "RESEARCH_MEMO_TARGET_CHARACTERS 不能大于 "
                "RESEARCH_MEMO_MAX_CHARACTERS"
            )
        return self


def get_settings() -> Settings:
    """Return validated application settings."""

    return Settings()
