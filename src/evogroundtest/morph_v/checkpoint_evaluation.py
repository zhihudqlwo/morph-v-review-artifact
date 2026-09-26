"""Label-late evaluation of MORPH-V library checkpoints.

All checkpoint prediction banks are authenticated and cross-checked before the
label file is opened.  Metrics are reported per carrier and as an unweighted
macro-average, preserving benchmark denominators instead of pooling examples.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Sequence, Tuple

from .autonomous_discovery import DiscoveryContractError
from .expanded_dsl import CARRIERS


CHECKPOINTS = ("L0", "L1", "L2", "L3")
_LABEL_KEYS = {"answer", "answer_index", "correct", "correctness", "gold", "label", "reward"}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reject_embedded_labels(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key in _LABEL_KEYS:
                raise DiscoveryContractError(
                    "checkpoint bank contains a label-bearing field: %s.%s"
                    % (path, raw_key)
                )
            _reject_embedded_labels(child, "%s.%s" % (path, raw_key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_embedded_labels(child, "%s[%d]" % (path, index))


def load_frozen_prediction_bank(path: Path) -> Dict[str, Dict[str, Any]]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("prediction bank or metadata is missing: %s" % path)
    rows = load_jsonl(path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    _reject_embedded_labels(rows)
    if metadata.get("outcomes_read") is not False:
        raise DiscoveryContractError("prediction metadata crossed the outcome boundary")
    if metadata.get("generation_complete") is not True:
        raise DiscoveryContractError("prediction bank is not frozen complete")
    if metadata.get("prediction_bank_sha256") != sha_file(path):
        raise DiscoveryContractError("prediction bank hash drift")
    if int(metadata.get("prediction_count", -1)) != len(rows):
        raise DiscoveryContractError("prediction bank count drift")
    values: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if set(row) - {
            "sample_id",
            "prediction",
            "status",
            "program_graph_id",
            "route",
            "outcomes_read",
        }:
            raise DiscoveryContractError("prediction bank contains undeclared fields")
        if row.get("outcomes_read") is not False:
            raise DiscoveryContractError("prediction row lacks outcome-hidden audit")
        sample_id = str(row["sample_id"])
        if sample_id in values:
            raise DiscoveryContractError("duplicate prediction sample ID")
        graph_id = str(row.get("program_graph_id") or "Direct")
        route = str(
            row.get("route")
            or ("direct" if graph_id.lower() == "direct" else "program")
        )
        if route not in {
            "direct",
            "direct_fallback",
            "program",
            "program_safe_gate",
        }:
            raise DiscoveryContractError("prediction row has an invalid route")
        # Reliability gating is deployment metadata on a program route.  It
        # remains program coverage for checkpoint-level reporting.
        route_class = "program" if route == "program_safe_gate" else route
        values[sample_id] = {
            "prediction": str(row["prediction"]),
            "status": str(row.get("status") or ""),
            "program_graph_id": graph_id,
            "route": route_class,
        }
    return values


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("empty percentile")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_macro_delta(
    correctness: Mapping[str, Mapping[str, Mapping[str, bool]]],
    video_ids: Mapping[str, Mapping[str, str]],
    checkpoint: str,
    baseline: str,
    seed: int,
    replicates: int,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        carrier_deltas = []
        for carrier in CARRIERS:
            current = correctness[checkpoint][carrier]
            parent = correctness[baseline][carrier]
            if set(current) != set(parent) or not current:
                raise DiscoveryContractError("checkpoint correctness alignment drift")
            clusters: Dict[str, list[str]] = {}
            for sample_id in sorted(current):
                clusters.setdefault(video_ids[carrier][sample_id], []).append(sample_id)
            names = sorted(clusters)
            sampled = [names[rng.randrange(len(names))] for _ in names]
            deltas = [
                float(current[sample_id]) - float(parent[sample_id])
                for name in sampled
                for sample_id in clusters[name]
            ]
            carrier_deltas.append(
                sum(deltas) / len(deltas)
            )
        values.append(100.0 * sum(carrier_deltas) / len(carrier_deltas))
    return _percentile(values, 0.025), _percentile(values, 0.975)


def _paired_transitions(
    current: Mapping[str, bool], baseline: Mapping[str, bool]
) -> Dict[str, int]:
    if set(current) != set(baseline):
        raise DiscoveryContractError("paired transition population drift")
    w2r = sum(not baseline[key] and current[key] for key in current)
    r2w = sum(baseline[key] and not current[key] for key in current)
    return {"w2r": w2r, "r2w": r2w, "net": w2r - r2w}


def _exact_mcnemar(w2r: int, r2w: int) -> float:
    discordant = w2r + r2w
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(w2r, r2w) + 1)
    ) / float(2**discordant)
    return min(1.0, 2.0 * tail)


def _holm_adjust(values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(values, key=lambda key: (values[key], key))
    adjusted: Dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * values[key]))
        adjusted[key] = running
    return adjusted


def evaluate_checkpoints(
    prediction_paths: Mapping[str, Mapping[str, Path]],
    label_path: Path,
    seed: int = 20260916,
    bootstrap_replicates: int = 10000,
) -> Dict[str, Any]:
    """Authenticate all banks, then perform one final label join."""
    if set(prediction_paths) != set(CHECKPOINTS):
        raise DiscoveryContractError("evaluation requires exactly L0--L3")
    banks: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    bank_hashes: Dict[str, Dict[str, str]] = {}
    for checkpoint in CHECKPOINTS:
        if set(prediction_paths[checkpoint]) != set(CARRIERS):
            raise DiscoveryContractError("checkpoint is missing a carrier bank")
        banks[checkpoint] = {}
        bank_hashes[checkpoint] = {}
        for carrier in CARRIERS:
            path = prediction_paths[checkpoint][carrier]
            banks[checkpoint][carrier] = load_frozen_prediction_bank(path)
            bank_hashes[checkpoint][carrier] = sha_file(path)

    for carrier in CARRIERS:
        id_sets = {
            frozenset(banks[checkpoint][carrier]) for checkpoint in CHECKPOINTS
        }
        if len(id_sets) != 1:
            raise DiscoveryContractError("checkpoint sample IDs drift within carrier")

    # Labels are intentionally opened only after every bank above passes.
    labels_rows = load_jsonl(label_path)
    labels: Dict[str, Dict[str, str]] = {carrier: {} for carrier in CARRIERS}
    video_ids: Dict[str, Dict[str, str]] = {carrier: {} for carrier in CARRIERS}
    for row in labels_rows:
        if set(row) - {"carrier", "sample_id", "video_id", "label"}:
            raise DiscoveryContractError("label file contains undeclared fields")
        carrier = str(row["carrier"])
        if carrier not in labels:
            raise DiscoveryContractError("label file contains an unknown carrier")
        sample_id = str(row["sample_id"])
        if sample_id in labels[carrier]:
            raise DiscoveryContractError("duplicate label sample ID")
        labels[carrier][sample_id] = str(row["label"])
        video_ids[carrier][sample_id] = str(row.get("video_id") or sample_id)
    for carrier in CARRIERS:
        expected = set(banks["L0"][carrier])
        if set(labels[carrier]) != expected:
            raise DiscoveryContractError("label/prediction population mismatch")

    correctness: Dict[str, Dict[str, Dict[str, bool]]] = {}
    metrics: Dict[str, Any] = {}
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        correctness[checkpoint] = {}
        carrier_metrics = {}
        for carrier in CARRIERS:
            ids = sorted(labels[carrier])
            flags = {
                sample_id: banks[checkpoint][carrier][sample_id]["prediction"]
                == labels[carrier][sample_id]
                for sample_id in ids
            }
            correctness[checkpoint][carrier] = flags
            covered = {
                sample_id
                for sample_id in ids
                if banks[checkpoint][carrier][sample_id]["route"] == "program"
            }
            prior_covered = (
                set()
                if checkpoint_index == 0
                else {
                    sample_id
                    for sample_id in ids
                    if banks[CHECKPOINTS[checkpoint_index - 1]][carrier][sample_id][
                        "route"
                    ]
                    == "program"
                }
            )
            carrier_metrics[carrier] = {
                "correct": sum(flags.values()),
                "rows": len(flags),
                "accuracy_percent": 100.0 * sum(flags.values()) / len(flags),
                "covered_by_program": len(covered),
                "coverage_percent": 100.0 * len(covered) / len(flags),
                "newly_covered_since_previous": len(covered - prior_covered),
                "technical_failures": sum(
                    banks[checkpoint][carrier][sample_id]["status"] != "ok"
                    for sample_id in ids
                ),
            }
        macro = sum(
            carrier_metrics[carrier]["accuracy_percent"] for carrier in CARRIERS
        ) / len(CARRIERS)
        macro_coverage = sum(
            carrier_metrics[carrier]["coverage_percent"] for carrier in CARRIERS
        ) / len(CARRIERS)
        metrics[checkpoint] = {
            "carriers": carrier_metrics,
            "macro_accuracy_percent": macro,
            "macro_coverage_percent": macro_coverage,
            "delta_vs_L0_percentage_points": 0.0,
        }
    baseline_macro = metrics["L0"]["macro_accuracy_percent"]
    for index, checkpoint in enumerate(CHECKPOINTS):
        delta = metrics[checkpoint]["macro_accuracy_percent"] - baseline_macro
        metrics[checkpoint]["delta_vs_L0_percentage_points"] = delta
        if checkpoint == "L0":
            metrics[checkpoint]["delta_vs_L0_bootstrap_ci95"] = [0.0, 0.0]
        else:
            ci = _bootstrap_macro_delta(
                correctness,
                video_ids,
                checkpoint,
                "L0",
                seed + index,
                bootstrap_replicates,
            )
            metrics[checkpoint]["delta_vs_L0_bootstrap_ci95"] = list(ci)

        aggregate_current = {
            "%s/%s" % (carrier, sample_id): flag
            for carrier in CARRIERS
            for sample_id, flag in correctness[checkpoint][carrier].items()
        }
        aggregate_baseline = {
            "%s/%s" % (carrier, sample_id): flag
            for carrier in CARRIERS
            for sample_id, flag in correctness["L0"][carrier].items()
        }
        transitions = _paired_transitions(aggregate_current, aggregate_baseline)
        metrics[checkpoint]["paired_vs_L0"] = {
            **transitions,
            "exact_mcnemar_p": _exact_mcnemar(
                transitions["w2r"], transitions["r2w"]
            ),
        }

    adjacent_tests: Dict[str, Dict[str, Any]] = {}
    raw_p_values: Dict[str, float] = {}
    for index in range(1, len(CHECKPOINTS)):
        current_name = CHECKPOINTS[index]
        parent_name = CHECKPOINTS[index - 1]
        aggregate_current = {
            "%s/%s" % (carrier, sample_id): flag
            for carrier in CARRIERS
            for sample_id, flag in correctness[current_name][carrier].items()
        }
        aggregate_parent = {
            "%s/%s" % (carrier, sample_id): flag
            for carrier in CARRIERS
            for sample_id, flag in correctness[parent_name][carrier].items()
        }
        transitions = _paired_transitions(aggregate_current, aggregate_parent)
        comparison = "%s_vs_%s" % (current_name, parent_name)
        raw_p_values[comparison] = _exact_mcnemar(
            transitions["w2r"], transitions["r2w"]
        )
        adjacent_tests[comparison] = {
            "delta_percentage_points": metrics[current_name][
                "macro_accuracy_percent"
            ]
            - metrics[parent_name]["macro_accuracy_percent"],
            "bootstrap_ci95": list(
                _bootstrap_macro_delta(
                    correctness,
                    video_ids,
                    current_name,
                    parent_name,
                    seed + 100 + index,
                    bootstrap_replicates,
                )
            ),
            **transitions,
            "exact_mcnemar_p": raw_p_values[comparison],
        }
    adjusted = _holm_adjust(raw_p_values)
    for comparison in adjacent_tests:
        adjacent_tests[comparison]["holm_adjusted_p"] = adjusted[comparison]
    adjacent = [
        adjacent_tests["%s_vs_%s" % (CHECKPOINTS[index], CHECKPOINTS[index - 1])][
            "delta_percentage_points"
        ]
        >= 0.0
        for index in range(1, len(CHECKPOINTS))
    ]
    return {
        "schema_version": 1,
        "status": "EVALUATED_FROZEN_CHECKPOINTS",
        "checkpoint_metrics": metrics,
        "growth_gain_percentage_points": metrics["L3"]["macro_accuracy_percent"]
        - baseline_macro,
        "nondecreasing_adjacent_checkpoints": all(adjacent),
        "adjacent_nondecrease_checks": adjacent,
        "adjacent_checkpoint_tests": adjacent_tests,
        "multiple_comparison_correction": "Holm over three adjacent exact McNemar tests",
        "macro_definition": "unweighted mean of interval, slot_set, and event_time accuracies",
        "uncertainty_definition": "paired bootstrap over video clusters within carrier, then unweighted carrier macro-average",
        "prediction_bank_sha256": bank_hashes,
        "label_file_sha256": sha_file(label_path),
        "labels_opened_after_all_prediction_banks_froze": True,
        "bootstrap_replicates": bootstrap_replicates,
    }
