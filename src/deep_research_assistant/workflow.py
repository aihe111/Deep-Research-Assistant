"""Application-level orchestration for the first research planning milestone."""

from pydantic import BaseModel

from deep_research_assistant.hy3_client import Hy3Client
from deep_research_assistant.intent_analyzer import IntentAnalyzer
from deep_research_assistant.models import ResearchIntent, ResearchOutline
from deep_research_assistant.outline_generator import OutlineGenerator


class ClarificationRequired(ValueError):
    """Raised when the workflow must wait for missing user parameters."""

    def __init__(self, intent: ResearchIntent) -> None:
        self.intent = intent
        super().__init__("调研意图需要用户补充")


class ResearchPlan(BaseModel):
    """The persisted result of the intent and outline stages."""

    original_request: str
    intent: ResearchIntent
    outline: ResearchOutline


class PlanningWorkflow:
    """Run intent analysis followed by outline generation."""

    def __init__(self, client: Hy3Client | None = None) -> None:
        model_client = client or Hy3Client()
        self.intent_analyzer = IntentAnalyzer(model_client)
        self.outline_generator = OutlineGenerator(model_client)

    def run(self, request: str) -> ResearchPlan:
        intent = self.analyze(request)
        if intent.clarification_questions:
            raise ClarificationRequired(intent)
        return self.build(request, intent)

    def analyze(self, request: str) -> ResearchIntent:
        """Analyze only; callers may collect clarification before continuing."""

        return self.intent_analyzer.analyze(request)

    def build(self, request: str, intent: ResearchIntent) -> ResearchPlan:
        """Generate an outline only after clarification is complete."""

        if intent.clarification_questions:
            raise ClarificationRequired(intent)
        outline = self.outline_generator.generate(intent)
        return ResearchPlan(original_request=request, intent=intent, outline=outline)
