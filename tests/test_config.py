from deep_research_assistant.config import Settings


def test_default_settings_are_valid() -> None:
    settings = Settings(_env_file=None)

    assert settings.hy3_base_url == "https://tokenhub.tencentmaas.com/v1"
    assert settings.hy3_model == "hy3"
    assert settings.hy3_reasoning_effort == "low"
    assert settings.hy3_timeout_seconds > 0
    assert settings.openalex_base_url == "https://api.openalex.org"
    assert settings.openalex_per_query == 30
    assert settings.tavily_base_url == "https://api.tavily.com"
    assert settings.tavily_hub_base_url == "https://tavily.sharyuke.com/api/proxy"
    assert settings.tavily_active_api_key == ""
    assert settings.database_url == "sqlite:///data/deep_research_assistant.db"
    assert settings.langgraph_api_url == "http://127.0.0.1:2024"
    assert settings.langgraph_assistant_id == "deep_research_assistant"


def test_no_think_is_a_valid_reasoning_effort() -> None:
    settings = Settings(_env_file=None, hy3_reasoning_effort="no_think")

    assert settings.hy3_reasoning_effort == "no_think"


def test_tavily_hub_key_takes_precedence_over_official_key() -> None:
    settings = Settings(
        _env_file=None,
        tavily_api_key="tvly-official",
        tavily_hub_api_key="thb-hub",
    )

    assert settings.tavily_active_api_key == "thb-hub"
    assert settings.tavily_active_base_url == "https://tavily.sharyuke.com/api/proxy"
    assert settings.tavily_provider == "hub"
