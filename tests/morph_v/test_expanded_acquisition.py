from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from evogroundtest.morph_v.autonomous_discovery import DiscoveryContractError
from evogroundtest.morph_v.expanded_acquisition import (
    PROCEDURES,
    AcquisitionCluster,
    AcquisitionLimits,
    build_proposal_packet,
    load_context_bank,
    run_acquisition,
)
from evogroundtest.morph_v.expanded_dsl import (
    HELD_OUT_SIGNATURES,
    build_default_registry,
    proposal_for_signature,
)


def context_for(carrier: str):
    if carrier == "interval":
        return {
            "carrier": carrier,
            "defect_present": True,
            "duration": 12.0,
            "frame_budget": 6,
            "lexical_intervals": [{"start": 3.0, "end": 5.0, "score": 0.74}],
            "semantic_intervals": [{"start": 4.0, "end": 6.0, "score": 0.94}],
            "shift_equivariance": 0.92,
            "answer_calls": 0,
        }
    if carrier == "slot_set":
        return {
            "carrier": carrier,
            "defect_present": True,
            "frame_budget": 3,
            "slot_scores": {
                "s1": [0.9, 0.1, 0.2, 0.3],
                "s2": [0.1, 0.9, 0.2, 0.3],
                "s3": [0.1, 0.2, 0.9, 0.3],
            },
            "negative_scores": {
                "s1": [0.05, 0.05, 0.05, 0.05],
                "s2": [0.05, 0.05, 0.05, 0.05],
                "s3": [0.05, 0.05, 0.05, 0.05],
            },
            "answer_calls": 0,
        }
    if carrier == "event_time":
        return {
            "carrier": carrier,
            "defect_present": True,
            "frame_budget": 8,
            "event_views": {
                "a": {
                    "e1": {"time": 1.0, "confidence": 0.90},
                    "e2": {"time": 4.0, "confidence": 0.87},
                    "e3": {"time": 7.0, "confidence": 0.85},
                },
                "b": {
                    "e1": {"time": 1.1, "confidence": 0.88},
                    "e2": {"time": 4.2, "confidence": 0.89},
                    "e3": {"time": 6.9, "confidence": 0.86},
                },
                "c": {
                    "e1": {"time": 0.9, "confidence": 0.86},
                    "e2": {"time": 4.1, "confidence": 0.88},
                    "e3": {"time": 7.1, "confidence": 0.84},
                },
            },
            "shuffled_relation_sha256": "0" * 64,
            "identical_visual_schedule": True,
            "answer_calls": 0,
        }
    raise AssertionError(carrier)


def clusters():
    order = ("interval", "slot_set", "event_time")
    return tuple(
        AcquisitionCluster(
            cluster_id="cluster-%s" % carrier,
            carrier=carrier,
            row_count=30 - index,
            summary={"pattern": "anonymous-%s" % carrier},
            examples=({"anonymous_id": "example-%d" % index},),
            contexts=(context_for(carrier), context_for(carrier)),
            source_sha256=(str(index + 1) * 64)[:64],
        )
        for index, carrier in enumerate(order)
    )


def target_proposer(packet):
    carrier = packet["residual_cluster"]["carrier"]
    signature = next(
        value for value in HELD_OUT_SIGNATURES.values() if value["carrier"] == carrier
    )
    return asdict(
        proposal_for_signature(
            carrier,
            signature["binder"],
            signature["transformation"],
            signature["objective"],
            suggested_name="NeutralGeneratedProgram",
        )
    )


def test_prompt_exposes_primitives_but_not_held_out_family_aliases() -> None:
    registry = build_default_registry()
    packet = build_proposal_packet(clusters()[0], registry, (), 1)
    wire = json.dumps(packet, ensure_ascii=False).lower()
    assert packet["primitive_registry"]["compatible_signature_count"] == 81
    for alias in HELD_OUT_SIGNATURES:
        assert alias.lower() not in wire


def test_model_driven_run_recovers_three_targets_and_saves_l0_l3(tmp_path) -> None:
    result = run_acquisition(
        "morph_v",
        clusters(),
        build_default_registry(),
        AcquisitionLimits(candidate_cap=1, proposal_attempt_cap=2),
        proposer=target_proposer,
        output_dir=tmp_path,
    )
    assert result["heldout_recovery_count"] == 3
    assert result["heldout_recovery_denominator"] == 3
    assert result["unique_composition_count"] == 3
    assert [path.name for path in sorted((tmp_path / "checkpoints").glob("*.json"))] == [
        "L0.json",
        "L1.json",
        "L2.json",
        "L3.json",
    ]


@pytest.mark.parametrize("procedure", PROCEDURES[1:])
def test_all_matched_controls_execute(procedure: str) -> None:
    kwargs = {"proposer": target_proposer} if procedure.startswith("without_") else {}
    result = run_acquisition(
        procedure,
        clusters(),
        build_default_registry(),
        AcquisitionLimits(candidate_cap=1, proposal_attempt_cap=2),
        **kwargs,
    )
    assert result["unique_composition_count"] == 3
    assert result["admission_rate"] == 1.0
    assert result["outcomes_read"] is False


def test_random_control_records_invalid_candidate_and_continues(monkeypatch) -> None:
    from evogroundtest.morph_v import expanded_acquisition as acquisition_module

    original = acquisition_module._execute_candidate
    calls = 0

    def fail_first(graph, registry, cluster, limits):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DiscoveryContractError("invalid structural trace")
        return original(graph, registry, cluster, limits)

    monkeypatch.setattr(acquisition_module, "_execute_candidate", fail_first)
    result = run_acquisition(
        "random_compatible_composition",
        clusters(),
        build_default_registry(),
        AcquisitionLimits(candidate_cap=4),
    )
    assert result["cycles"][0]["attempts"][0]["status"] == "REJECTED"
    assert result["unique_composition_count"] == 3


def test_without_persistence_feedback_hides_prior_graphs() -> None:
    observed_sizes = []

    def proposer(packet):
        observed_sizes.append(packet["current_library"]["stored_graph_count"])
        return target_proposer(packet)

    result = run_acquisition(
        "without_persistence_feedback",
        clusters(),
        build_default_registry(),
        AcquisitionLimits(candidate_cap=1),
        proposer=proposer,
    )
    assert observed_sizes == [0, 0, 0]
    assert result["unique_composition_count"] == 3


def test_frozen_carrier_order_overrides_cluster_recurrence() -> None:
    observed = []

    def proposer(packet):
        observed.append(packet["residual_cluster"]["carrier"])
        return target_proposer(packet)

    result = run_acquisition(
        "morph_v",
        clusters(),
        build_default_registry(),
        AcquisitionLimits(candidate_cap=1),
        proposer=proposer,
        carrier_order=("event_time", "slot_set", "interval"),
    )

    assert observed == ["event_time", "slot_set", "interval"]
    assert result["carrier_order"] == ["event_time", "slot_set", "interval"]


def test_context_bank_rejects_outcomes(tmp_path) -> None:
    row = {
        "cluster_id": "cluster-interval",
        "carrier": "interval",
        "row_count": 1,
        "summary": {"pattern": "anonymous"},
        "examples": [],
        "source_sha256": "a" * 64,
        "context": {**context_for("interval"), "gold_label": "A"},
        "outcomes_read": False,
    }
    path = tmp_path / "bank.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DiscoveryContractError, match="outcome-bearing"):
        load_context_bank(path)


def test_context_bank_derives_bounded_anonymous_examples(tmp_path) -> None:
    row = {
        "cluster_id": "cluster-slot-set",
        "carrier": "slot_set",
        "row_count": 1,
        "summary": {"pattern": "anonymous"},
        "examples": [],
        "source_sha256": "a" * 64,
        "context": {
            **context_for("slot_set"),
            "sample_id": "private-sample-id",
            "video_relpath": "private/video.mp4",
        },
        "outcomes_read": False,
    }
    path = tmp_path / "bank.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = load_context_bank(path)
    packet = build_proposal_packet(loaded[0], build_default_registry(), (), 1)
    examples = packet["residual_cluster"]["anonymous_examples"]

    assert examples == [
        {
            "anonymous_id": "slot_set-example-1",
            "slot_count": 3,
            "lattice_size": 4,
            "mean_top_slot_score": 0.9,
            "mean_top_contrast": 0.85,
            "frame_budget": 3,
        }
    ]
    wire = json.dumps(packet, ensure_ascii=False)
    assert "private-sample-id" not in wire
    assert "private/video.mp4" not in wire
