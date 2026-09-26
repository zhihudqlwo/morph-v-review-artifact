#!/usr/bin/env python3
"""Offline reproduction from released records; never calls a model endpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evogroundtest.morph_v.checkpoint_evaluation import evaluate_checkpoints
from evogroundtest.morph_v.expanded_acquisition import (
    AcquisitionLimits, ProposalEnvelope, load_context_bank, run_acquisition,
)
from evogroundtest.morph_v.expanded_dsl import (
    CARRIERS, ExpandedRelationLibrary, build_default_registry, compatible_signatures,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def compare(actual, expected, path="metrics"):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), "Field mismatch at " + path)
        for key in expected:
            compare(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), "Length mismatch at " + path)
        for index, value in enumerate(expected):
            compare(actual[index], value, "%s[%d]" % (path, index))
    elif isinstance(expected, float):
        require(math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-12), "Numeric mismatch at " + path)
    else:
        require(actual == expected, "Value mismatch at " + path)


class FrozenReplies:
    def __init__(self):
        self.rows = [json.loads(line) for line in (DATA / "acquisition/responses.jsonl").read_text().splitlines() if line.strip()]
        self.position = 0

    def __call__(self, packet):
        require(self.position < len(self.rows), "Frozen reply ledger exhausted")
        response = self.rows[self.position]
        self.position += 1
        return ProposalEnvelope(response, {
            "source": "frozen_model_response_ledger",
            "model": "Qwen3.6-27B",
            "ledger_position": self.position,
        })


def acquisition(output):
    expected = json.loads((DATA / "acquisition/expected.json").read_text())
    replies = FrozenReplies()
    result = run_acquisition(
        "morph_v", load_context_bank(DATA / "acquisition/contexts.jsonl"),
        build_default_registry(), AcquisitionLimits(**expected["limits"]),
        proposer=replies, output_dir=output / "acquisition",
        carrier_order=expected["carrier_order"],
    )
    attempts = [item for cycle in result["cycles"] for item in cycle["attempts"]]
    valid = sum(item["status"] == "EXECUTED_VALID_CANDIDATE" for item in attempts)
    require(replies.position == len(replies.rows) == expected["proposal_attempts"], "Proposal-count mismatch")
    require(valid == expected["structurally_valid_candidates"], "Valid-candidate mismatch")
    require(result["unique_composition_count"] == expected["unique_compositions"], "Stored-program mismatch")
    require(result["heldout_recovery_count"] == expected["exact_reference_matches"], "Reference-match mismatch")
    require([c["selected_graph_id"] for c in result["cycles"]] == expected["selected_graph_ids"], "Selected-graph mismatch")
    require(result["registry_digest"] == expected["registry_digest"], "Registry mismatch")
    print("PASS: acquisition replay: 18 recorded proposals, 10 valid candidates, 3 stored graphs, 0 exact reference matches; no model calls.")


def checkpoints(output):
    registry = build_default_registry()
    require(len(compatible_signatures()) == 81, "Composition-space mismatch")
    clusters = load_context_bank(DATA / "acquisition/contexts.jsonl")
    contexts = {cluster.carrier: cluster.contexts for cluster in clusters}
    report = {}
    for index in range(4):
        name = "L%d" % index
        library = ExpandedRelationLibrary.load(DATA / "checkpoints" / (name + ".json"), registry)
        require(library.size_with_direct == index + 1, "Library-size mismatch")
        executed = 0
        for graph in library.graphs:
            for context in contexts[graph.carrier]:
                record = library.execute(graph.graph_id, context)
                require(not record.outcomes_read and all(record.invariant_checks.values()), "Invalid reloaded execution")
                executed += 1
        report[name] = {"entries_including_direct": library.size_with_direct, "structural_executions": executed}
    save(output / "checkpoint_reload.json", report)
    print("PASS: 81 compatible signatures; original L0--L3 libraries reload and execute through the generic registry.")


def growth(output):
    paths = {"L%d" % i: {carrier: DATA / "growth" / ("L%d" % i) / carrier / "predictions.jsonl" for carrier in CARRIERS} for i in range(4)}
    result = evaluate_checkpoints(paths, DATA / "growth/labels.jsonl", seed=20260916, bootstrap_replicates=10000)
    comparable = {key: value for key, value in result.items() if key not in {"prediction_bank_sha256", "label_file_sha256"}}
    expected = json.loads((DATA / "growth/expected_metrics.json").read_text())
    compare(comparable, expected)
    save(output / "growth_evaluation.json", result)
    print("Checkpoint  Interval  Slot-set  Event-time  Macro   Gain vs L0")
    for name, item in sorted(result["checkpoint_metrics"].items()):
        accuracy = [item["carriers"][carrier]["accuracy_percent"] for carrier in CARRIERS]
        print("%-10s  %7.2f  %8.2f  %10.2f  %5.2f  %+10.2f" % (name, *accuracy, item["macro_accuracy_percent"], item["delta_vs_L0_percentage_points"]))
    print("PASS: every released metric matches the frozen report, including 10,000-draw video-cluster intervals and Holm-adjusted tests.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "acquisition", "checkpoints", "growth"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reproduction")
    args = parser.parse_args()
    stages = {"acquisition": acquisition, "checkpoints": checkpoints, "growth": growth}
    for name, function in stages.items():
        if args.stage in ("all", name):
            function(args.output_dir.resolve())
    print("No training, video downloads, GPU, endpoint, or API key was used.")


if __name__ == "__main__":
    main()
