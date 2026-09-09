from types import SimpleNamespace

from typer.testing import CliRunner

import deep_research_assistant.cli as cli_module
from deep_research_assistant.models import OutlineSection, ResearchIntent, ResearchOutline
from deep_research_assistant.workflow import ResearchPlan


class FakeInteractiveWorkflow:
    analyze_calls = 0
    build_calls = 0

    def analyze(self, request: str) -> ResearchIntent:
        self.__class__.analyze_calls += 1
        if self.analyze_calls == 1:
            return ResearchIntent(
                topic="RAG",
                goal="形成综述",
                clarification_questions=[
                    "期望最终报告大约多少字？",
                    "报告的目标读者是谁？",
                    "希望覆盖什么时间范围？",
                    "报告重点关注哪些方向？",
                ],
            )
        assert "3000字" in request
        assert "开发者" in request
        return ResearchIntent(
            topic="RAG",
            goal="形成综述",
            audience="开发者",
            focus_areas=["检索质量"],
            start_year=2023,
            end_year=2026,
            target_word_count=3000,
        )

    def build(self, request: str, intent: ResearchIntent) -> ResearchPlan:
        self.__class__.build_calls += 1
        assert self.analyze_calls == 2
        return ResearchPlan(
            original_request=request,
            intent=intent,
            outline=ResearchOutline(
                title="RAG 调研",
                thesis="梳理 RAG",
                sections=[
                    OutlineSection(
                        section_id=f"S{index}",
                        title=f"章节 {index}",
                        objective="回答问题",
                        research_questions=["核心问题是什么？"],
                    )
                    for index in range(1, 4)
                ],
            ),
        )


def test_plan_command_waits_for_answers_before_outline(monkeypatch, tmp_path) -> None:
    FakeInteractiveWorkflow.analyze_calls = 0
    FakeInteractiveWorkflow.build_calls = 0
    monkeypatch.setattr(
        cli_module,
        "get_settings",
        lambda: SimpleNamespace(hy3_api_key="configured"),
    )
    monkeypatch.setattr(cli_module, "PlanningWorkflow", FakeInteractiveWorkflow)
    output = tmp_path / "plan.json"

    result = CliRunner().invoke(
        cli_module.app,
        ["plan", "调研 RAG", "--output", str(output)],
        input="3000字\n开发者\n2023-2026年\n检索质量\n",
    )

    assert result.exit_code == 0, result.output
    assert FakeInteractiveWorkflow.analyze_calls == 2
    assert FakeInteractiveWorkflow.build_calls == 1
    assert output.is_file()
