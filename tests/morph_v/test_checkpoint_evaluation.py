from __future__ import annotations

import json

import pytest

from evogroundtest.morph_v.autonomous_discovery import DiscoveryContractError
from evogroundtest.morph_v.checkpoint_evaluation import (
    CHECKPOINTS,
    evaluate_checkpoints,
    sha_file,
)
from evogroundtest.morph_v.expanded_dsl import CARRIERS


def write_bank(path, predictions):
    rows = [
        {
            "sample_id": sample_id,
            "prediction": prediction,
            "status": "ok",
            "program_graph_id": "graph-test",
            "outcomes_read": False,
        }
        for sample_id, prediction in sorted(predictions.items())
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "prediction_bank_sha256": sha_file(path),
        "prediction_count": len(rows),
        "generation_complete": True,
        "outcomes_read": False,
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )


def frozen_fixture(tmp_path):
    paths = {checkpoint: {} for checkpoint in CHECKPOINTS}
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        for carrier_index, carrier in enumerate(CARRIERS):
            correct = carrier_index < checkpoint_index
            values = {"s1": "A", "s2": "B" if correct else "A"}
            path = tmp_path / checkpoint / carrier / "predictions.jsonl"
            write_bank(path, values)
            paths[checkpoint][carrier] = path
    labels = tmp_path / "LABELS.jsonl"
    labels.write_text(
        "".join(
            json.dumps({"carrier": carrier, "sample_id": sample_id, "label": label})
            + "\n"
            for carrier in CARRIERS
            for sample_id, label in (("s1", "A"), ("s2", "B"))
        ),
        encoding="utf-8",
    )
    return paths, labels


def test_checkpoint_evaluation_preserves_carriers_and_macro_average(tmp_path) -> None:
    paths, labels = frozen_fixture(tmp_path)
    result = evaluate_checkpoints(paths, labels, bootstrap_replicates=200)
    metrics = result["checkpoint_metrics"]
    assert metrics["L0"]["macro_accuracy_percent"] == 50.0
    assert metrics["L3"]["macro_accuracy_percent"] == 100.0
    assert result["growth_gain_percentage_points"] == 50.0
    assert result["nondecreasing_adjacent_checkpoints"] is True
    assert result["labels_opened_after_all_prediction_banks_froze"] is True
    for checkpoint in CHECKPOINTS:
        assert set(metrics[checkpoint]["carriers"]) == set(CARRIERS)
        assert all(
            metrics[checkpoint]["carriers"][carrier]["rows"] == 2
            for carrier in CARRIERS
        )


def test_prediction_hash_drift_stops_before_label_join(tmp_path) -> None:
    paths, labels = frozen_fixture(tmp_path)
    path = paths["L2"]["slot_set"]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DiscoveryContractError, match="hash drift"):
        evaluate_checkpoints(paths, labels, bootstrap_replicates=10)


def test_prediction_bank_cannot_embed_labels(tmp_path) -> None:
    paths, labels = frozen_fixture(tmp_path)
    path = paths["L1"]["interval"]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[0]["gold"] = "A"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
    meta["prediction_bank_sha256"] = sha_file(path)
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta) + "\n")
    with pytest.raises(DiscoveryContractError, match="label-bearing"):
        evaluate_checkpoints(paths, labels, bootstrap_replicates=10)


def test_checkpoint_report_includes_coverage_transitions_and_cluster_tests(tmp_path) -> None:
    paths, labels = frozen_fixture(tmp_path)
    label_rows = [
        json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()
    ]
    for row in label_rows:
        row["video_id"] = "video-" + row["sample_id"]
    labels.write_text(
        "".join(json.dumps(row) + "\n" for row in label_rows), encoding="utf-8"
    )
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        for carrier_index, carrier in enumerate(CARRIERS):
            path = paths[checkpoint][carrier]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                covered = carrier_index < checkpoint_index
                row["route"] = (
                    "program_safe_gate"
                    if covered and checkpoint == "L3"
                    else ("program" if covered else "direct")
                )
                row["program_graph_id"] = "graph-test" if covered else "Direct"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            meta = json.loads(meta_path.read_text())
            meta["prediction_bank_sha256"] = sha_file(path)
            meta_path.write_text(json.dumps(meta) + "\n")

    result = evaluate_checkpoints(paths, labels, bootstrap_replicates=200)
    assert result["checkpoint_metrics"]["L0"]["macro_coverage_percent"] == 0.0
    assert result["checkpoint_metrics"]["L3"]["macro_coverage_percent"] == 100.0
    assert result["checkpoint_metrics"]["L3"]["paired_vs_L0"] == {
        "w2r": 3,
        "r2w": 0,
        "net": 3,
        "exact_mcnemar_p": 0.25,
    }
    assert set(result["adjacent_checkpoint_tests"]) == {
        "L1_vs_L0",
        "L2_vs_L1",
        "L3_vs_L2",
    }
    assert all(
        "holm_adjusted_p" in value
        for value in result["adjacent_checkpoint_tests"].values()
    )
    assert "video clusters" in result["uncertainty_definition"]
