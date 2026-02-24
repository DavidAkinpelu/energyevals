from pathlib import Path

from energbench.evaluation.config import load_eval_config


def test_load_eval_config_defaults_to_gpt5_mini(tmp_path: Path) -> None:
    config_path = tmp_path / "eval_config.yaml"
    config_path.write_text("judge:\n  provider: openai\n")

    config = load_eval_config(config_path, base_path=tmp_path)

    assert config.judge.model == "gpt-5-mini"


def test_load_eval_config_compare_and_reasoning_effort(tmp_path: Path) -> None:
    config_path = tmp_path / "eval_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "compare: true",
                "judge:",
                "  provider: openai",
                "  model: gpt-4o",
                "  reasoning_effort: high",
            ]
        )
    )

    config = load_eval_config(config_path, base_path=tmp_path)

    assert config.compare is True
    assert config.judge.reasoning_effort == "high"
