from types import SimpleNamespace

import pytest

from scripts.run_benchmark_generation import (
    BenchmarkCase,
    build_parser,
    load_manifest,
    normalize_example,
    normalize_examples,
    pending_cases,
    save_manifest,
    select_labeled_examples,
)


def _example(
    number: int,
    *,
    inputs: dict | None = None,
    metadata: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"example-{number}",
        inputs=inputs if inputs is not None else {"question": f"问题 {number}"},
        metadata=(
            metadata
            if metadata is not None
            else {
                "case_id": f"case-{number:02d}",
                "domain": "technology",
                "difficulty": "medium",
            }
        ),
    )


def _case(number: int) -> BenchmarkCase:
    return normalize_example(_example(number))  # type: ignore[arg-type]


def test_normalize_example_accepts_labels_in_metadata_or_inputs() -> None:
    metadata_case = normalize_example(_example(1))  # type: ignore[arg-type]
    input_case = normalize_example(
        _example(
            2,
            inputs={
                "case_id": "input-02",
                "domain": "finance",
                "difficulty": "hard",
                "question": "金融问题",
            },
            metadata={},
        )
    )  # type: ignore[arg-type]

    assert metadata_case.case_id == "case-01"
    assert input_case.case_id == "input-02"
    assert input_case.domain == "finance"
    assert input_case.question == "金融问题"


def test_normalize_examples_rejects_duplicate_case_ids() -> None:
    first = _example(1)
    second = _example(
        2,
        metadata={
            "case_id": "case-01",
            "domain": "academic",
            "difficulty": "easy",
        },
    )

    with pytest.raises(ValueError, match="case_id 必须唯一"):
        normalize_examples([first, second])  # type: ignore[list-item]


def test_select_labeled_examples_ignores_legacy_rows() -> None:
    labeled = _example(1)
    legacy = _example(2, metadata={})

    selected, ignored = select_labeled_examples(  # type: ignore[list-item]
        [legacy, labeled]
    )

    assert selected == [labeled]
    assert ignored == 1


def test_pending_cases_skips_complete_and_retries_failed() -> None:
    cases = [_case(number) for number in range(1, 9)]
    manifest = {
        "cases": {
            "case-01": {"status": "complete"},
            "case-02": {"status": "failed"},
        }
    }

    selected = pending_cases(cases, manifest, batch_size=5)

    assert [case.case_id for case in selected] == [
        "case-02",
        "case-03",
        "case-04",
        "case-05",
        "case-06",
    ]


def test_parser_accepts_one_forced_case() -> None:
    args = build_parser().parse_args(
        ["benchmark_dataset", "--case-id", "finance_005", "--force"]
    )

    assert args.dataset == "benchmark_dataset"
    assert args.case_id == "finance_005"
    assert args.force is True


def test_manifest_round_trip_preserves_checkpoint(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    manifest = load_manifest(path, "bird", "dataset-1")
    manifest["cases"]["case-01"] = {"status": "complete"}
    save_manifest(path, manifest)

    loaded = load_manifest(path, "bird", "dataset-1")

    assert loaded["cases"]["case-01"]["status"] == "complete"
    assert not path.with_name("manifest.json.tmp").exists()
