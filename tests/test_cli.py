import json

import pytest

from st_intel.cli import main, summarize
from st_intel.config import Config


def write_result(tmp_path, method, seed, value, **overrides):
    config = Config(method=method, seed=seed, **overrides).as_dict()
    path = tmp_path / f"{method}-{seed}.json"
    path.write_text(
        json.dumps(
            dict(
                method=method,
                seed=seed,
                config=config,
                evaluation=dict(
                    environment_seed=seed + 100000, network=dict(throughput_mbps=value)
                ),
            )
        ),
        encoding="utf-8",
    )
    return path


def test_summary_uses_seed_uncertainty_and_paired_differences(tmp_path):
    paths = [
        write_result(tmp_path, method, seed, base + seed)
        for method, base in [("st-intel", 5), ("ddqn", 3)]
        for seed in [1, 2, 3]
    ]
    summary = summarize(paths, "throughput_mbps")
    assert summary["metrics"]["throughput_mbps"]["st-intel"]["statistics"]["mean"] == 7
    assert summary["metrics"]["throughput_mbps"]["st-intel"]["statistics"]["std"] == 1
    paired = summary["paired_st_intel_minus_control"]["throughput_mbps"]["ddqn"]
    assert paired["seeds"] == [1, 2, 3]
    assert paired["difference"]["mean"] == 2
    assert paired["p_value"] == 0


def test_summary_rejects_duplicate_and_mismatched_scenarios(tmp_path):
    first = write_result(tmp_path, "st-intel", 1, 5)
    with pytest.raises(ValueError, match="duplicate"):
        summarize([first, first], "throughput_mbps")
    second = write_result(tmp_path, "ddqn", 1, 3, scenario="unstable")
    with pytest.raises(ValueError, match="different experiment"):
        summarize([first, second], "throughput_mbps")


def test_summary_no_uncertainty_for_single_seed_and_no_data(tmp_path):
    first = write_result(tmp_path, "st-intel", 1, 5)
    second = write_result(tmp_path, "ddqn", 1, None)
    summary = summarize([first, second], "throughput_mbps")
    assert summary["metrics"]["throughput_mbps"]["st-intel"]["statistics"]["ci95"] is None
    assert summary["metrics"]["throughput_mbps"]["ddqn"]["statistics"] is None
    assert summary["paired_st_intel_minus_control"]["throughput_mbps"]["ddqn"] is None


def test_cli_writes_summary(tmp_path):
    path = write_result(tmp_path, "st-intel", 1, 5)
    output = tmp_path / "summary.json"
    main(["summarize", str(path), "--metric", "throughput_mbps", "--output", str(output)])
    assert json.loads(output.read_text())["phase"] == "frozen_policy_evaluation"


def test_summary_rejects_changed_pairing_seed(tmp_path):
    path = write_result(tmp_path, "st-intel", 1, 5)
    record = json.loads(path.read_text())
    record["evaluation"]["environment_seed"] = 999
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="evaluation seed"):
        summarize([path], "throughput_mbps")


def test_supplied_configs_are_valid():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("smoke", "paper_reference"):
        Config.load(root / "configs" / f"{name}.json")
