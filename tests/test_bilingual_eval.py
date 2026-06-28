import json

from scripts.compare_evaluations import collect_rows, render_markdown
from scripts.eval_bilingual import (
    EvaluationRun,
    centered_chat_scores,
    generation_quality,
    ngram_diversity,
    parse_named_paths,
)


def minimal_manifest(source="sft", tag="demo", step=1):
    return {
        "schema_version": 1,
        "run_id": f"{source}-{tag}-{step}",
        "created_at": "2026-06-21T00:00:00+00:00",
        "git": {"commit": "abc", "branch": "test", "dirty": False},
        "model": {"source": source, "tag": tag, "step": step},
        "evaluation": {"device_type": "cpu"},
    }


def test_ngram_and_generation_quality():
    assert ngram_diversity("aaaa", 2) == 1 / 3
    quality = generation_quality([
        {"prompt": "Explain spring.", "response": "Explain spring. Spring is warm."},
        {"prompt": "Explain math.", "response": ""},
    ])
    assert quality["responses"] == 2
    assert quality["prompt_copy_responses"] == 1
    assert quality["empty_responses"] == 1


def test_chatcore_aggregates():
    results = {
        "ARC-Easy": 0.25,
        "ARC-Challenge": 0.25,
        "MMLU": 0.25,
        "GSM8K": 0.0,
        "HumanEval": 0.0,
        "SpellingBee": 0.9,
    }
    aggregates = centered_chat_scores(results)
    assert aggregates["chatcore"] == 0.15
    assert aggregates["chatcore_without_spellingbee"] == 0.0


def test_parse_named_paths(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert parse_named_paths([f"english={data_dir}"]) == {
        "english": str(data_dir),
    }


def test_evaluation_run_persists_and_resumes(tmp_path):
    output_dir = tmp_path / "run"
    manifest = minimal_manifest()
    run = EvaluationRun(str(output_dir), manifest, resume=False)
    run.save_metric("language", {
        "summary": {
            "zh_response_rate": 1.0,
            "en_response_rate": 1.0,
            "replacement_character_responses": 0,
        },
        "generation_quality": {
            "zh": {
                "mean_characters": 10,
                "mean_distinct_2": 0.5,
                "mean_distinct_3": 0.6,
                "prompt_copy_responses": 0,
                "empty_responses": 0,
            },
            "en": {
                "mean_characters": 10,
                "mean_distinct_2": 0.5,
                "mean_distinct_3": 0.6,
                "prompt_copy_responses": 0,
                "empty_responses": 0,
            },
        },
    })
    run.finish()

    resumed = EvaluationRun(str(output_dir), manifest, resume=True)
    assert resumed.status["state"] == "completed"
    assert resumed.metrics["language"]["summary"]["zh_response_rate"] == 1.0
    assert (output_dir / "summary.md").exists()


def test_evaluation_run_rejects_model_mismatch(tmp_path):
    output_dir = tmp_path / "run"
    EvaluationRun(str(output_dir), minimal_manifest(tag="one"), resume=False)
    try:
        EvaluationRun(str(output_dir), minimal_manifest(tag="two"), resume=True)
    except ValueError as error:
        assert "Resume model mismatch" in str(error)
    else:
        raise AssertionError("Expected resume with another model to fail")


def test_comparison_collects_metrics(tmp_path):
    run_path = tmp_path / "run"
    run_path.mkdir()
    run = {
        "path": str(run_path),
        "manifest": minimal_manifest(),
        "status": {"state": "completed"},
        "metrics": {
            "chat": {
                "tasks": {
                    "ARC-Easy": {"status": "completed", "accuracy": 0.5},
                },
                "aggregates": {"partial_chatcore": 1 / 3},
            },
        },
    }
    rows = collect_rows([run])
    assert rows[0]["chat"]["ARC-Easy"] == 0.5
    markdown = render_markdown(rows)
    assert "ARC-E" in markdown
    assert "50.00%" in markdown

    # Ensure rows remain JSON serializable for comparison.json.
    json.dumps(rows)
