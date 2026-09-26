"""Outcome-hidden acquisition loop for the expanded MORPH-V DSL.

The runner is transport-agnostic: a proposer callback may call a remote model,
replay a frozen response ledger, or be replaced by a matched control.  All
procedures share the same candidate and structural-execution caps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .autonomous_discovery import (
    DiscoveryContractError,
    assert_no_family_name_leakage,
    assert_outcome_hidden,
)
from .expanded_dsl import (
    APPLICABILITY_BY_CARRIER,
    BINDERS_BY_CARRIER,
    CARRIERS,
    DEFECT_BY_CARRIER,
    HELD_OUT_SIGNATURES,
    OBJECTIVES_BY_CARRIER,
    TRANSFORMATIONS_BY_CARRIER,
    BatchAdmissionRecord,
    CompiledGraph,
    ExpandedProposal,
    ExpandedRelationLibrary,
    PrimitiveRegistry,
    aggregate_execution_records,
    compile_expanded_proposal,
    execute_compiled_graph,
    proposal_for_signature,
)


PROCEDURES = (
    "morph_v",
    "fixed_carrier_template",
    "random_compatible_composition",
    "without_structural_ranking",
    "without_persistence_feedback",
)


@dataclass(frozen=True)
class AcquisitionCluster:
    cluster_id: str
    carrier: str
    row_count: int
    summary: Mapping[str, Any]
    examples: Tuple[Mapping[str, Any], ...]
    contexts: Tuple[Mapping[str, Any], ...]
    source_sha256: str

    def __post_init__(self) -> None:
        if self.carrier not in CARRIERS:
            raise DiscoveryContractError("acquisition cluster has an unknown carrier")
        if self.row_count < 1 or not self.contexts:
            raise DiscoveryContractError("acquisition cluster is empty")
        for context in self.contexts:
            assert_outcome_hidden(context)
            if context.get("carrier") != self.carrier:
                raise DiscoveryContractError("cluster context carrier drift")

    def prompt_view(self) -> Dict[str, Any]:
        value = {
            "cluster_id": self.cluster_id,
            "carrier": self.carrier,
            "recurrence_count": self.row_count,
            "structural_summary": dict(self.summary),
            "anonymous_examples": [dict(item) for item in self.examples],
        }
        assert_outcome_hidden(value)
        assert_no_family_name_leakage(value)
        return value


@dataclass(frozen=True)
class AcquisitionLimits:
    minimum_recurrence: int = 1
    proposal_attempt_cap: int = 6
    candidate_cap: int = 4
    structural_execution_cap: int = 64
    seed: int = 20260916

    def __post_init__(self) -> None:
        if min(
            self.minimum_recurrence,
            self.proposal_attempt_cap,
            self.candidate_cap,
            self.structural_execution_cap,
        ) < 1:
            raise DiscoveryContractError("all acquisition limits must be positive")


@dataclass(frozen=True)
class CandidateResult:
    graph: CompiledGraph
    admission: BatchAdmissionRecord
    proposal_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "admission": self.admission.to_dict(),
            "proposal_index": self.proposal_index,
        }


@dataclass(frozen=True)
class ProposalEnvelope:
    proposal: Mapping[str, Any]
    transport: Mapping[str, Any]


Proposer = Callable[[Mapping[str, Any]], Any]


def expanded_proposal_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "suggested_name",
            "applicability",
            "defect",
            "binder",
            "transformation",
            "invariants",
            "objective",
            "rationale",
        ],
        "properties": {
            "suggested_name": {"type": "string", "minLength": 1, "maxLength": 64},
            "applicability": {"type": "string"},
            "defect": {"type": "string"},
            "binder": {"type": "string"},
            "transformation": {"type": "string"},
            "invariants": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "objective": {"type": "string"},
            "rationale": {"type": "string", "minLength": 1},
        },
    }


def registry_prompt_inventory(registry: PrimitiveRegistry) -> Dict[str, Any]:
    roles = (
        "applicability",
        "defect",
        "binder",
        "transformation",
        "invariant",
        "objective",
    )
    return {
        "semantic_version": registry.semantic_version,
        "registry_digest": registry.digest,
        "primitives": {
            role: [spec.public_dict() for spec in registry.inventory(role)]
            for role in roles
        },
        "compatible_signature_count": 81,
    }


def library_prompt_view(graphs: Sequence[CompiledGraph]) -> Dict[str, Any]:
    return {
        "fallback": "Direct",
        "stored_graph_count": len(graphs),
        "stored_signatures": [
            {
                "graph_id": graph.graph_id,
                "carrier": graph.carrier,
                "applicability": graph.applicability,
                "binder": graph.binder,
                "transformation": graph.transformation,
                "objective": graph.objective,
            }
            for graph in graphs
        ],
    }


def build_proposal_packet(
    cluster: AcquisitionCluster,
    registry: PrimitiveRegistry,
    visible_graphs: Sequence[CompiledGraph],
    candidate_index: int,
    compiler_feedback: Optional[str] = None,
    rejected_signatures: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    packet: Dict[str, Any] = {
        "task": "compose one executable observation program for an uncovered recurrent structure",
        "candidate_index": candidate_index,
        "residual_cluster": cluster.prompt_view(),
        "current_library": library_prompt_view(visible_graphs),
        "primitive_registry": registry_prompt_inventory(registry),
        "requirements": [
            "select only registered primitives whose carrier and typed interfaces compose",
            "include all global and carrier-specific invariants",
            "do not reproduce a complete stored signature",
            "do not repeat a rejected primitive signature; revise at least one primitive in response to structural feedback",
            "use a neutral proposed name; identity is determined only from the executable graph",
            "do not infer, request, or use answers, labels, predictions, correctness, reward, or benchmark identity",
        ],
        "response_schema": expanded_proposal_json_schema(),
    }
    if compiler_feedback:
        packet["compiler_feedback"] = compiler_feedback[:1000]
    if rejected_signatures:
        packet["rejected_signatures"] = [dict(item) for item in rejected_signatures]
    assert_outcome_hidden(packet)
    assert_no_family_name_leakage(packet)
    return packet


def _signature(graph: CompiledGraph) -> Tuple[str, str, str, str]:
    return (graph.carrier, graph.binder, graph.transformation, graph.objective)


def recovered_alias(graph: CompiledGraph) -> Optional[str]:
    matches = [
        alias
        for alias, expected in HELD_OUT_SIGNATURES.items()
        if _signature(graph)
        == (
            expected["carrier"],
            expected["binder"],
            expected["transformation"],
            expected["objective"],
        )
    ]
    if len(matches) > 1:
        raise DiscoveryContractError("one graph matches multiple evaluation aliases")
    return matches[0] if matches else None


def _execute_candidate(
    graph: CompiledGraph,
    registry: PrimitiveRegistry,
    cluster: AcquisitionCluster,
    limits: AcquisitionLimits,
) -> BatchAdmissionRecord:
    contexts = cluster.contexts[: limits.structural_execution_cap]
    records = [
        execute_compiled_graph(graph, registry, context)
        for context in contexts
    ]
    return aggregate_execution_records(records)


def _seeded_order(seed: int, cluster_id: str, graph_id: str) -> str:
    from hashlib import sha256

    return sha256((str(seed) + "|" + cluster_id + "|" + graph_id).encode()).hexdigest()


def _control_proposals(
    procedure: str,
    cluster: AcquisitionCluster,
    limits: AcquisitionLimits,
) -> List[ExpandedProposal]:
    carrier = cluster.carrier
    if procedure == "fixed_carrier_template":
        return [
            proposal_for_signature(
                carrier,
                BINDERS_BY_CARRIER[carrier][0],
                TRANSFORMATIONS_BY_CARRIER[carrier][0],
                OBJECTIVES_BY_CARRIER[carrier][0],
                suggested_name="FixedCarrierTemplate",
            )
        ]
    if procedure != "random_compatible_composition":
        raise DiscoveryContractError("procedure does not define control proposals")
    candidates = [
        (binder, transformation, objective)
        for binder in BINDERS_BY_CARRIER[carrier]
        for transformation in TRANSFORMATIONS_BY_CARRIER[carrier]
        for objective in OBJECTIVES_BY_CARRIER[carrier]
    ]
    salt = int(_seeded_order(limits.seed, cluster.cluster_id, carrier)[:16], 16)
    rng = random.Random(salt)
    rng.shuffle(candidates)
    return [
        proposal_for_signature(
            carrier,
            binder,
            transformation,
            objective,
            suggested_name="RandomCompatible%02d" % (index + 1),
        )
        for index, (binder, transformation, objective) in enumerate(
            candidates[: limits.candidate_cap]
        )
    ]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _anonymous_context_example(
    context: Mapping[str, Any], index: int
) -> Dict[str, Any]:
    carrier = str(context["carrier"])
    value: Dict[str, Any] = {"anonymous_id": "%s-example-%d" % (carrier, index)}
    if carrier == "interval":
        lexical = list(context.get("lexical_intervals") or [])
        semantic = list(context.get("semantic_intervals") or [])
        top_lexical = max((float(item["score"]) for item in lexical), default=0.0)
        top_semantic = max((float(item["score"]) for item in semantic), default=0.0)
        value.update(
            {
                "lexical_candidate_count": len(lexical),
                "semantic_candidate_count": len(semantic),
                "lexical_specificity": round(
                    top_lexical - float(context.get("lexical_decoy_score", 0.0)), 6
                ),
                "semantic_specificity": round(
                    top_semantic - float(context.get("semantic_decoy_score", 0.0)), 6
                ),
                "frame_budget": int(context["frame_budget"]),
            }
        )
    elif carrier == "slot_set":
        scores = context.get("slot_scores") or {}
        negatives = context.get("negative_scores") or {}
        tops = [max(float(item) for item in row) for row in scores.values()]
        contrasts = []
        for slot, row in scores.items():
            negative_row = list(negatives.get(slot) or [])
            contrasts.append(
                max(
                    float(score)
                    - float(negative_row[offset] if offset < len(negative_row) else 0.0)
                    for offset, score in enumerate(row)
                )
            )
        value.update(
            {
                "slot_count": len(scores),
                "lattice_size": max((len(row) for row in scores.values()), default=0),
                "mean_top_slot_score": round(sum(tops) / len(tops), 6) if tops else 0.0,
                "mean_top_contrast": round(sum(contrasts) / len(contrasts), 6)
                if contrasts
                else 0.0,
                "frame_budget": int(context["frame_budget"]),
            }
        )
    elif carrier == "event_time":
        views = context.get("event_views") or {}
        names = sorted({name for events in views.values() for name in events})
        confidences = [
            float(item["confidence"])
            for events in views.values()
            for item in events.values()
        ]
        spreads = []
        for name in names:
            times = [
                float(events[name]["time"])
                for events in views.values()
                if name in events
            ]
            if times:
                spreads.append(max(times) - min(times))
        value.update(
            {
                "view_count": len(views),
                "event_count": len(names),
                "mean_confidence": round(
                    sum(confidences) / len(confidences), 6
                )
                if confidences
                else 0.0,
                "mean_cross_view_spread_seconds": round(
                    sum(spreads) / len(spreads), 6
                )
                if spreads
                else 0.0,
                "frame_budget": int(context["frame_budget"]),
            }
        )
    else:
        raise DiscoveryContractError("cannot summarize an unknown carrier")
    return value


def run_acquisition(
    procedure: str,
    clusters: Sequence[AcquisitionCluster],
    registry: PrimitiveRegistry,
    limits: AcquisitionLimits,
    proposer: Optional[Proposer] = None,
    output_dir: Optional[Path] = None,
    carrier_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run one matched acquisition procedure over fixed structural clusters."""
    started = time.monotonic()
    if procedure not in PROCEDURES:
        raise DiscoveryContractError("unknown acquisition procedure")
    model_driven = procedure in {
        "morph_v",
        "without_structural_ranking",
        "without_persistence_feedback",
    }
    if model_driven and proposer is None:
        raise DiscoveryContractError("model-driven acquisition requires a proposer")

    if carrier_order is not None:
        normalized_order = tuple(str(item) for item in carrier_order)
        if len(normalized_order) != len(set(normalized_order)) or set(
            normalized_order
        ) != set(CARRIERS):
            raise DiscoveryContractError(
                "carrier order must contain each registered carrier exactly once"
            )
        carrier_position = {
            carrier: index for index, carrier in enumerate(normalized_order)
        }
        ordering = lambda item: (
            carrier_position[item.carrier],
            -item.row_count,
            item.cluster_id,
        )
    else:
        normalized_order = None
        ordering = lambda item: (-item.row_count, item.cluster_id)
    eligible = sorted(
        [cluster for cluster in clusters if cluster.row_count >= limits.minimum_recurrence],
        key=ordering,
    )
    if not eligible:
        raise DiscoveryContractError("no recurrent acquisition cluster")
    if len({cluster.cluster_id for cluster in eligible}) != len(eligible):
        raise DiscoveryContractError("duplicate acquisition cluster IDs")

    visible_graphs: List[CompiledGraph] = []
    selected_pairs: List[Tuple[CompiledGraph, BatchAdmissionRecord]] = []
    cycles: List[Dict[str, Any]] = []
    processed: set = set()

    while len(processed) < len(eligible):
        covered_carriers = {graph.carrier for graph in visible_graphs}
        remaining = [
            cluster
            for cluster in eligible
            if cluster.cluster_id not in processed
            and (
                procedure == "without_persistence_feedback"
                or cluster.carrier not in covered_carriers
            )
        ]
        if not remaining:
            break
        cluster = remaining[0]
        processed.add(cluster.cluster_id)
        visible_size_before = len(visible_graphs)
        attempts: List[Dict[str, Any]] = []
        candidates: List[CandidateResult] = []
        seen_graphs: set = set()
        rejected_signatures: List[Dict[str, Any]] = []

        if model_driven:
            feedback = None
            assert proposer is not None
            for attempt_index in range(limits.proposal_attempt_cap):
                packet = build_proposal_packet(
                    cluster,
                    registry,
                    visible_graphs,
                    len(candidates) + 1,
                    feedback,
                    rejected_signatures,
                )
                raw_response = proposer(packet)
                if isinstance(raw_response, ProposalEnvelope):
                    response = dict(raw_response.proposal)
                    transport = dict(raw_response.transport)
                else:
                    response = dict(raw_response)
                    transport = {"source": "in_process_proposer"}
                assert_outcome_hidden(response)
                assert_outcome_hidden(transport)
                attempt: Dict[str, Any] = {
                    "attempt": attempt_index + 1,
                    "packet": packet,
                    "response": response,
                    "transport": transport,
                    "outcomes_read": False,
                }
                try:
                    proposal = ExpandedProposal.from_mapping(response)
                    graph = compile_expanded_proposal(proposal, registry, cluster.carrier)
                    if graph.graph_id in seen_graphs or any(
                        _signature(graph) == _signature(stored) for stored in visible_graphs
                    ):
                        raise DiscoveryContractError("duplicate complete signature")
                    admission = _execute_candidate(graph, registry, cluster, limits)
                    seen_graphs.add(graph.graph_id)
                    candidates.append(CandidateResult(graph, admission, attempt_index + 1))
                    attempt["status"] = "EXECUTED_VALID_CANDIDATE"
                    attempt["graph_id"] = graph.graph_id
                    attempt["admission_sha256"] = admission.admission_sha256
                    feedback = None
                except (DiscoveryContractError, KeyError, TypeError, ValueError) as exc:
                    feedback = "%s: %s" % (type(exc).__name__, exc)
                    attempt["status"] = "REJECTED"
                    attempt["compiler_feedback"] = feedback[:1000]
                    signature = {
                        key: response.get(key)
                        for key in ("binder", "transformation", "objective")
                    }
                    if all(signature.values()) and signature not in rejected_signatures:
                        rejected_signatures.append(signature)
                attempts.append(attempt)
                if len(candidates) >= limits.candidate_cap:
                    break
        else:
            proposals = _control_proposals(procedure, cluster, limits)
            for proposal_index, proposal in enumerate(proposals, 1):
                attempt: Dict[str, Any] = {
                    "attempt": proposal_index,
                    "outcomes_read": False,
                }
                try:
                    graph = compile_expanded_proposal(proposal, registry, cluster.carrier)
                    admission = _execute_candidate(graph, registry, cluster, limits)
                    candidates.append(CandidateResult(graph, admission, proposal_index))
                    attempt["status"] = "EXECUTED_VALID_CANDIDATE"
                    attempt["graph_id"] = graph.graph_id
                    attempt["admission_sha256"] = admission.admission_sha256
                except (DiscoveryContractError, KeyError, TypeError, ValueError) as exc:
                    attempt["status"] = "REJECTED"
                    attempt["compiler_feedback"] = "%s: %s" % (type(exc).__name__, exc)
                attempts.append(attempt)

        if not candidates:
            if output_dir is not None:
                _atomic_json(
                    Path(output_dir).resolve() / "FAILED_ACQUISITION.json",
                    {
                        "schema_version": 1,
                        "procedure": procedure,
                        "cluster_id": cluster.cluster_id,
                        "carrier": cluster.carrier,
                        "attempts": attempts,
                        "visible_stored_graphs_before": visible_size_before,
                        "outcomes_read": False,
                    },
                )
            raise DiscoveryContractError(
                "procedure %s produced no valid candidate for %s"
                % (procedure, cluster.cluster_id)
            )
        if procedure == "without_structural_ranking":
            selected = sorted(
                candidates,
                key=lambda item: _seeded_order(
                    limits.seed, cluster.cluster_id, item.graph.graph_id
                ),
            )[0]
            selection_policy = "fixed_seeded_valid_order"
        elif procedure == "fixed_carrier_template":
            selected = candidates[0]
            selection_policy = "preassigned_carrier_template"
        else:
            selected = sorted(
                candidates,
                key=lambda item: (
                    -item.admission.mean_objective_score,
                    item.graph.graph_id,
                ),
            )[0]
            selection_policy = "outcome_hidden_structural_ranking"

        selected_pairs.append((selected.graph, selected.admission))
        if procedure != "without_persistence_feedback":
            visible_graphs.append(selected.graph)
        cycle = {
            "cycle": len(cycles) + 1,
            "cluster_id": cluster.cluster_id,
            "carrier": cluster.carrier,
            "recurrence_count": cluster.row_count,
            "attempts": attempts,
            "valid_candidates": [candidate.to_dict() for candidate in candidates],
            "selection_policy": selection_policy,
            "selected_graph_id": selected.graph.graph_id,
            "selected_signature": {
                "carrier": selected.graph.carrier,
                "binder": selected.graph.binder,
                "transformation": selected.graph.transformation,
                "objective": selected.graph.objective,
            },
            "posthoc_recovery_alias": recovered_alias(selected.graph),
            "visible_stored_graphs_before": visible_size_before,
            "visible_stored_graphs_after": len(visible_graphs),
            "outcomes_read": False,
        }
        cycles.append(cycle)

    final_library = ExpandedRelationLibrary(registry)
    checkpoint_payloads = [final_library.to_dict()]
    for graph, admission in selected_pairs:
        if graph.graph_id in {item.graph_id for item in final_library.graphs}:
            continue
        final_library.admit(graph, admission)
        checkpoint_payloads.append(final_library.to_dict())

    recovered = [
        alias
        for alias in (recovered_alias(graph) for graph in final_library.graphs)
        if alias is not None
    ]
    all_attempts = [attempt for cycle in cycles for attempt in cycle["attempts"]]
    model_attempts = [
        attempt
        for attempt in all_attempts
        if (attempt.get("transport") or {}).get("source")
        not in {None, "in_process_proposer", "matched_control"}
    ]
    usage_totals: Dict[str, float] = {}
    for attempt in model_attempts:
        for key, value in ((attempt.get("transport") or {}).get("usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage_totals[str(key)] = usage_totals.get(str(key), 0.0) + float(value)
    result = {
        "schema_version": 1,
        "procedure": procedure,
        "limits": asdict(limits),
        "carrier_order": list(normalized_order) if normalized_order else None,
        "registry_version": registry.semantic_version,
        "registry_digest": registry.digest,
        "cycles": cycles,
        "compile_rate": sum(
            attempt["status"] == "EXECUTED_VALID_CANDIDATE"
            for cycle in cycles
            for attempt in cycle["attempts"]
        )
        / max(1, sum(len(cycle["attempts"]) for cycle in cycles)),
        "admission_rate": len(selected_pairs) / max(1, len(eligible)),
        "unique_composition_count": len(final_library.graphs),
        "posthoc_recovered_aliases": sorted(recovered),
        "heldout_recovery_count": len(set(recovered)),
        "heldout_recovery_denominator": len(HELD_OUT_SIGNATURES),
        "acquisition_cost": {
            "proposal_calls": len(model_attempts),
            "proposal_latency_seconds": sum(
                float((attempt.get("transport") or {}).get("latency_seconds", 0.0))
                for attempt in model_attempts
            ),
            "proposal_latency_recorded_calls": sum(
                "latency_seconds" in (attempt.get("transport") or {})
                for attempt in model_attempts
            ),
            "proposal_usage": usage_totals,
            "compiled_valid_candidates": sum(
                len(cycle["valid_candidates"]) for cycle in cycles
            ),
            "structural_executions": sum(
                int(candidate["admission"]["population_size"])
                for cycle in cycles
                for candidate in cycle["valid_candidates"]
            ),
            "total_wall_seconds": time.monotonic() - started,
        },
        "final_library": final_library.to_dict(),
        "outcomes_read": False,
    }
    if output_dir is not None:
        output_dir = output_dir.resolve()
        _atomic_json(output_dir / "ACQUISITION_RESULT.json", result)
        for index, checkpoint in enumerate(checkpoint_payloads):
            _atomic_json(output_dir / "checkpoints" / ("L%d.json" % index), checkpoint)
    return result


def load_context_bank(path: Path) -> Tuple[AcquisitionCluster, ...]:
    """Load a frozen, outcome-hidden JSONL carrier bank for acquisition."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert_outcome_hidden(rows)
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    metadata: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("outcomes_read") is not False:
            raise DiscoveryContractError("context-bank row lacks outcome-hidden audit")
        cluster_id = str(row["cluster_id"])
        grouped.setdefault(cluster_id, []).append(dict(row["context"]))
        current = {
            "carrier": str(row["carrier"]),
            "row_count": int(row["row_count"]),
            "summary": dict(row["summary"]),
            "examples": tuple(dict(item) for item in row.get("examples") or []),
            "source_sha256": str(row["source_sha256"]),
        }
        if cluster_id in metadata and metadata[cluster_id] != current:
            raise DiscoveryContractError("context-bank cluster metadata drift")
        metadata[cluster_id] = current
    clusters = []
    for cluster_id in sorted(grouped):
        value = metadata[cluster_id]
        examples = tuple(value["examples"]) or tuple(
            _anonymous_context_example(context, index + 1)
            for index, context in enumerate(grouped[cluster_id][:3])
        )
        clusters.append(
            AcquisitionCluster(
                cluster_id=cluster_id,
                carrier=str(value["carrier"]),
                row_count=int(value["row_count"]),
                summary=dict(value["summary"]),
                examples=examples,
                contexts=tuple(grouped[cluster_id]),
                source_sha256=str(value["source_sha256"]),
            )
        )
    return tuple(clusters)
