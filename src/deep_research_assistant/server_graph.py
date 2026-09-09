"""LangGraph Server entry point for Deep-Research-Assistant."""

import os

from deep_research_assistant.artifact_store import ArtifactStore
from deep_research_assistant.config import get_settings
from deep_research_assistant.conversation_store import ConversationStore
from deep_research_assistant.deep_research_graph import build_deep_research_graph
from deep_research_assistant.research_execution import answer_follow_up

settings = get_settings()
if settings.langsmith_tracing:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
artifacts = ArtifactStore(settings.database_url)
events = ConversationStore(settings.database_url)

# LangGraph Server supplies Thread/Run persistence. Application artifacts remain
# in DATABASE_URL so reports and the custom Web UI can share them with the graph.
graph = build_deep_research_graph(
    settings,
    artifacts,
    event_store=events,
    followup_handler=lambda thread_id: answer_follow_up(settings, thread_id),
)
