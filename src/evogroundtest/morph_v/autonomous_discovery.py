"""Autonomous, outcome-hidden discovery of executable MORPH-V relation genes.

The proposer composes a relation from typed primitives.  This module owns the
deterministic boundary: packet sanitization, type checking, program compilation,
structural admission, and persistent-library updates.  It never calls an answer
model and never consumes benchmark outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class DiscoveryContractError(ValueError):
    """Raised when autonomous discovery leaves its typed, outcome-hidden contract."""


APPLICABILITY_PRIMITIVES = (
    "explicit_text_span_requests_concurrent_visual_state",
    "multiple_semantic_observations_must_all_be_found",
    "each_option_is_an_ordered_sequence_of_retrievable_events",
)

APPLICABILITY_TYPES: Mapping[str, str] = {
    "explicit_text_span_requests_concurrent_visual_state": "interval",
    "multiple_semantic_observations_must_all_be_found": "slot_set",
    "each_option_is_an_ordered_sequence_of_retrievable_events": "event_time_matrix",
}

DEFECT_PRIMITIVES = (
    "localized_anchor_does_not_receive_local_observation_budget",
    "independent_retrieval_can_starve_a_required_evidence_slot",
    "independent_event_times_are_not_composed_into_candidate_relations",
)

DEFECT_TYPES: Mapping[str, str] = {
    "localized_anchor_does_not_receive_local_observation_budget": "interval",
    "independent_retrieval_can_starve_a_required_evidence_slot": "slot_set",
    "independent_event_times_are_not_composed_into_candidate_relations": "event_time_matrix",
}

BINDER_PRIMITIVES: Mapping[str, str] = {
    "align_text_span_to_timestamped_transcript": "interval",
    "compile_question_into_semantic_evidence_slots": "slot_set",
    "retrieve_each_candidate_event_time_independently": "event_time_matrix",
}

REPAIR_PRIMITIVES: Mapping[str, str] = {
    "reallocate_fixed_local_frames_around_bound_interval": "interval",
    "balance_fixed_frame_budget_across_bound_slots": "slot_set",
    "serialize_candidate_pairwise_temporal_contrasts": "event_time_matrix",
}

OBJECTIVE_PRIMITIVES: Mapping[str, str] = {
    "maximize_binding_specificity_then_shift_equivariance": "interval",
    "maximize_minimum_slot_coverage_then_semantic_support": "slot_set",
    "maximize_cross_view_order_agreement_then_confident_margin": "event_time_matrix",
}

INVARIANT_PRIMITIVES = (
    "one_answer_call",
    "fixed_frame_budget",
    "no_outcome_fields",
    "deterministic_replay",
    "bound_interval_capture",
    "temporal_shift_equivariance",
    "cross_video_binding_specificity",
    "every_required_slot_covered",
    "balanced_slot_allocation",
    "identical_visual_schedule",
    "finite_relation_values",
    "shuffled_order_changes_relation_trace",
)

BASE_INVARIANTS = {
    "one_answer_call",
    "fixed_frame_budget",
    "no_outcome_fields",
    "deterministic_replay",
}

TYPE_INVARIANTS: Mapping[str, frozenset] = {
    "interval": frozenset(
        {
            "bound_interval_capture",
            "temporal_shift_equivariance",
            "cross_video_binding_specificity",
        }
    ),
    "slot_set": frozenset(
        {"every_required_slot_covered", "balanced_slot_allocation"}
    ),
    "event_time_matrix": frozenset(
        {
            "identical_visual_schedule",
            "finite_relation_values",
            "shuffled_order_changes_relation_trace",
        }
    ),
}

CANDIDATE_GRAMMARS: Mapping[str, Tuple[str, ...]] = {
    "reallocate_fixed_local_frames_around_bound_interval": (
        "preserve_parent_schedule",
        "bind_with_lexical_transcript_match",
        "bind_with_semantic_transcript_match",
        "matched_random_interval",
    ),
    "balance_fixed_frame_budget_across_bound_slots": (
        "uniform_parent",
        "single_query_ranker",
        "independent_slot_topk",
        "balanced_slot_cover",
        "matched_random_frames",
    ),
    "serialize_candidate_pairwise_temporal_contrasts": (
        "uniform_parent",
        "raw_event_times",
        "shuffled_pairwise_relations",
        "ordered_pairwise_relations",
    ),
}

SELECTED_EXECUTORS: Mapping[str, Tuple[str, str]] = {
    "reallocate_fixed_local_frames_around_bound_interval": (
        "bind_with_semantic_transcript_match",
        "semantic_child",
    ),
    "balance_fixed_frame_budget_across_bound_slots": (
        "balanced_slot_cover",
        "SlotCover48Global16",
    ),
    "serialize_candidate_pairwise_temporal_contrasts": (
        "ordered_pairwise_relations",
        "OrderContrastTableUniform64",
    ),
}

_OUTCOME_KEY_TOKENS = (
    "answer",
    "correct",
    "gold",
    "label",
    "outcome",
    "prediction",
    "reward",
    "target",
)
_FALSE_AUDIT_KEYS = {
    "answer_calls",
    "choices_accessed",
    "confirmation_accessed",
    "outcomes_read",
    "target_outcomes_used",
}
_FAMILY_MARKERS = ("quotebind", "evidencesetbind", "ordercontrastbind")


def canonical_sha(value: Any) -> str:
    wire = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def assert_outcome_hidden(value: Any, path: str = "root") -> None:
    """Reject outcome-bearing fields recursively while allowing explicit false audits."""
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            child_path = "%s.%s" % (path, raw_key)
            if any(token in key for token in _OUTCOME_KEY_TOKENS):
                if key in _FALSE_AUDIT_KEYS and child in (False, 0, None):
                    continue
                raise DiscoveryContractError(
                    "outcome-bearing field crossed discovery boundary: %s" % child_path
                )
            assert_outcome_hidden(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_outcome_hidden(child, "%s[%d]" % (path, index))


def assert_no_family_name_leakage(value: Any) -> None:
    lowered = json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    leaked = [marker for marker in _FAMILY_MARKERS if marker in lowered]
    if leaked:
        raise DiscoveryContractError(
            "proposal packet leaks held-out family name(s): %s" % ", ".join(leaked)
        )


def _clean_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value).strip()).strip("-")
    if not name:
        raise DiscoveryContractError("proposal must provide a non-empty neutral name")
    if any(marker in name.lower() for marker in _FAMILY_MARKERS):
        raise DiscoveryContractError("proposal name copies a held-out family name")
    return name[:64]


@dataclass(frozen=True)
class RelationProposal:
    suggested_name: str
    applicability: str
    defect: str
    binder: str
    repair: str
    invariants: Tuple[str, ...]
    objective: str
    rationale: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RelationProposal":
        required = {
            "suggested_name",
            "applicability",
            "defect",
            "binder",
            "repair",
            "invariants",
            "objective",
            "rationale",
        }
        if set(value) != required:
            raise DiscoveryContractError(
                "proposal fields must be exactly %s" % sorted(required)
            )
        assert_outcome_hidden(value)
        proposal = cls(
            suggested_name=_clean_name(str(value["suggested_name"])),
            applicability=str(value["applicability"]),
            defect=str(value["defect"]),
            binder=str(value["binder"]),
            repair=str(value["repair"]),
            invariants=tuple(str(item) for item in value["invariants"]),
            objective=str(value["objective"]),
            rationale=str(value["rationale"]).strip(),
        )
        proposal.validate()
        return proposal

    def validate(self) -> None:
        if self.applicability not in APPLICABILITY_PRIMITIVES:
            raise DiscoveryContractError("unknown applicability primitive")
        if self.defect not in DEFECT_PRIMITIVES:
            raise DiscoveryContractError("unknown defect primitive")
        if self.binder not in BINDER_PRIMITIVES:
            raise DiscoveryContractError("unknown binder primitive")
        if self.repair not in REPAIR_PRIMITIVES:
            raise DiscoveryContractError("unknown repair primitive")
        if self.objective not in OBJECTIVE_PRIMITIVES:
            raise DiscoveryContractError("unknown objective primitive")
        if not self.rationale:
            raise DiscoveryContractError("proposal rationale is empty")
        if len(set(self.invariants)) != len(self.invariants):
            raise DiscoveryContractError("proposal invariants must be unique")
        if not set(self.invariants).issubset(INVARIANT_PRIMITIVES):
            raise DiscoveryContractError("proposal contains an unknown invariant")


@dataclass(frozen=True)
class ResidualCluster:
    cluster_id: str
    applicability_key: str
    row_count: int
    summary: Mapping[str, Any]
    examples: Tuple[Mapping[str, Any], ...]
    source_sha256: str

    def prompt_view(self) -> Dict[str, Any]:
        value = {
            "cluster_id": self.cluster_id,
            "recurrence_count": self.row_count,
            "structural_summary": dict(self.summary),
            "anonymous_examples": [dict(example) for example in self.examples],
        }
        assert_outcome_hidden(value)
        assert_no_family_name_leakage(value)
        return value


@dataclass(frozen=True)
class CompiledRelationGene:
    gene_id: str
    proposed_name: str
    applicability: str
    defect: str
    binder: str
    repair: str
    objective: str
    invariants: Tuple[str, ...]
    candidates: Tuple[str, ...]
    program_nodes: Tuple[Mapping[str, Any], ...]
    proposal_sha256: str
    selected_candidate: Optional[str] = None
    executor_id: Optional[str] = None
    admission_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compile_proposal(
    proposal: RelationProposal, cluster: ResidualCluster
) -> CompiledRelationGene:
    """Type-check and compile ``(A,D,G,V,J)`` into a replayable program graph."""
    proposal.validate()
    if proposal.applicability != cluster.applicability_key:
        raise DiscoveryContractError("proposal does not explain the triggering residual cluster")
    applicability_type = APPLICABILITY_TYPES[proposal.applicability]
    defect_type = DEFECT_TYPES[proposal.defect]
    binder_type = BINDER_PRIMITIVES[proposal.binder]
    repair_type = REPAIR_PRIMITIVES[proposal.repair]
    objective_type = OBJECTIVE_PRIMITIVES[proposal.objective]
    if len(
        {
            applicability_type,
            defect_type,
            binder_type,
            repair_type,
            objective_type,
        }
    ) != 1:
        raise DiscoveryContractError(
            "A, D, binder, G and J have incompatible trace types"
        )
    required_invariants = BASE_INVARIANTS | set(TYPE_INVARIANTS[binder_type])
    if not required_invariants.issubset(proposal.invariants):
        missing = sorted(required_invariants - set(proposal.invariants))
        raise DiscoveryContractError("proposal omits required invariants: %s" % missing)
    proposal_payload = asdict(proposal)
    proposal_sha = canonical_sha(proposal_payload)
    gene_id = "relation-gene-%s" % canonical_sha(
        {
            "applicability": proposal.applicability,
            "defect": proposal.defect,
            "binder": proposal.binder,
            "repair": proposal.repair,
            "objective": proposal.objective,
            "invariants": sorted(proposal.invariants),
        }
    )[:16]
    nodes: Tuple[Mapping[str, Any], ...] = (
        {"op": "DETECT", "primitive": proposal.applicability},
        {"op": "DIAGNOSE", "primitive": proposal.defect},
        {"op": "BIND", "primitive": proposal.binder, "output_type": binder_type},
        {"op": "MUTATE", "primitive": proposal.repair, "input_type": repair_type},
        {"op": "VERIFY", "invariants": sorted(proposal.invariants)},
        {"op": "SELECT", "objective": proposal.objective, "input_type": objective_type},
        {"op": "PERSIST", "key": gene_id},
    )
    return CompiledRelationGene(
        gene_id=gene_id,
        proposed_name=proposal.suggested_name,
        applicability=proposal.applicability,
        defect=proposal.defect,
        binder=proposal.binder,
        repair=proposal.repair,
        objective=proposal.objective,
        invariants=tuple(sorted(proposal.invariants)),
        candidates=CANDIDATE_GRAMMARS[proposal.repair],
        program_nodes=nodes,
        proposal_sha256=proposal_sha,
    )


def _all_true(value: Mapping[str, Any]) -> bool:
    return bool(value) and all(item is True for item in value.values())


def admit_from_structural_certificate(
    gene: CompiledRelationGene,
    certificate: Mapping[str, Any],
    certificate_sha256: str,
) -> CompiledRelationGene:
    """Admit a compiled gene using only its pre-answer structural certificate."""
    if len(certificate_sha256) != 64:
        raise DiscoveryContractError("structural certificate SHA256 is malformed")
    repair = gene.repair
    selected_candidate, expected_executor = SELECTED_EXECUTORS[repair]
    if repair == "reallocate_fixed_local_frames_around_bound_interval":
        if (
            certificate.get("outcomes_read") is not False
            or certificate.get("target_outcomes_used") is not False
            or certificate.get("choices_accessed") is not False
            or certificate.get("confirmation_accessed") is not False
            or certificate.get("status") != "PASS_LABEL_FREE_EVOLUTION"
            or certificate.get("promoted_program") != expected_executor
        ):
            raise DiscoveryContractError("interval certificate fails outcome-hidden promotion")
        fitness = (certificate.get("candidate_fitness") or {}).get(expected_executor) or {}
        if fitness.get("passed") is not True or not _all_true(fitness.get("checks") or {}):
            raise DiscoveryContractError("interval certificate fails structural fitness")
    elif repair == "balance_fixed_frame_budget_across_bound_slots":
        if (
            certificate.get("outcomes_read") is not False
            or certificate.get("status") != "SELECTED"
            or certificate.get("selected_program") != expected_executor
        ):
            raise DiscoveryContractError("slot-set certificate fails structural selection")
    elif repair == "serialize_candidate_pairwise_temporal_contrasts":
        if (
            certificate.get("outcomes_read") is not False
            or int(certificate.get("answer_calls", -1)) != 0
            or certificate.get("status") != "ADMITTED"
            or certificate.get("selected_program") != expected_executor
            or not _all_true(certificate.get("checks") or {})
        ):
            raise DiscoveryContractError("event-relation certificate fails structural admission")
    else:
        raise DiscoveryContractError("no structural verifier for repair primitive")
    if selected_candidate not in gene.candidates:
        raise DiscoveryContractError("selected candidate is absent from compiled grammar")
    admission_sha = canonical_sha(
        {
            "gene_id": gene.gene_id,
            "certificate_sha256": certificate_sha256,
            "selected_candidate": selected_candidate,
            "executor_id": expected_executor,
            "outcomes_read": False,
        }
    )
    return replace(
        gene,
        selected_candidate=selected_candidate,
        executor_id=expected_executor,
        admission_sha256=admission_sha,
    )


class PersistentRelationLibrary:
    """Deterministic relation-gene library with Direct as the initial fallback."""

    def __init__(self, genes: Optional[Iterable[CompiledRelationGene]] = None):
        self._genes: List[CompiledRelationGene] = list(genes or [])
        self._validate()

    def _validate(self) -> None:
        if len({gene.gene_id for gene in self._genes}) != len(self._genes):
            raise DiscoveryContractError("relation library contains duplicate gene IDs")
        if len({gene.applicability for gene in self._genes}) != len(self._genes):
            raise DiscoveryContractError("relation library contains duplicate applicability")
        if any(gene.admission_sha256 is None for gene in self._genes):
            raise DiscoveryContractError("only structurally admitted genes may persist")

    @property
    def size_with_direct(self) -> int:
        return 1 + len(self._genes)

    @property
    def genes(self) -> Tuple[CompiledRelationGene, ...]:
        return tuple(self._genes)

    def covers(self, cluster: ResidualCluster) -> bool:
        return any(gene.applicability == cluster.applicability_key for gene in self._genes)

    def admit(self, gene: CompiledRelationGene) -> None:
        if gene.admission_sha256 is None:
            raise DiscoveryContractError("cannot persist an unadmitted gene")
        if any(existing.gene_id == gene.gene_id for existing in self._genes):
            raise DiscoveryContractError("gene already exists")
        if any(existing.applicability == gene.applicability for existing in self._genes):
            raise DiscoveryContractError("applicability is already covered")
        self._genes.append(gene)
        self._validate()

    def select_next(self, clusters: Sequence[ResidualCluster], minimum_recurrence: int) -> Optional[ResidualCluster]:
        eligible = [
            cluster
            for cluster in clusters
            if cluster.row_count >= minimum_recurrence and not self.covers(cluster)
        ]
        if not eligible:
            return None
        return sorted(eligible, key=lambda item: (-item.row_count, item.cluster_id))[0]

    def proposer_view(self) -> Dict[str, Any]:
        return {
            "fallback": "Direct",
            "stored_gene_count": len(self._genes),
            "stored_signatures": [
                {
                    "gene_id": gene.gene_id,
                    "applicability": gene.applicability,
                    "binder": gene.binder,
                    "repair": gene.repair,
                }
                for gene in self._genes
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "fallback": "Direct",
            "size_with_direct": self.size_with_direct,
            "genes": [gene.to_dict() for gene in self._genes],
            "outcomes_read": False,
        }


def proposal_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "suggested_name",
            "applicability",
            "defect",
            "binder",
            "repair",
            "invariants",
            "objective",
            "rationale",
        ],
        "properties": {
            "suggested_name": {"type": "string", "minLength": 1, "maxLength": 64},
            "applicability": {"enum": list(APPLICABILITY_PRIMITIVES)},
            "defect": {"enum": list(DEFECT_PRIMITIVES)},
            "binder": {"enum": list(BINDER_PRIMITIVES)},
            "repair": {"enum": list(REPAIR_PRIMITIVES)},
            "invariants": {
                "type": "array",
                "items": {"enum": list(INVARIANT_PRIMITIVES)},
                "minItems": 6,
                "maxItems": 7,
            },
            "objective": {"enum": list(OBJECTIVE_PRIMITIVES)},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 360},
        },
        "additionalProperties": False,
    }


__all__ = [
    "APPLICABILITY_PRIMITIVES",
    "APPLICABILITY_TYPES",
    "BINDER_PRIMITIVES",
    "CompiledRelationGene",
    "DEFECT_PRIMITIVES",
    "DEFECT_TYPES",
    "DiscoveryContractError",
    "INVARIANT_PRIMITIVES",
    "OBJECTIVE_PRIMITIVES",
    "PersistentRelationLibrary",
    "REPAIR_PRIMITIVES",
    "RelationProposal",
    "ResidualCluster",
    "admit_from_structural_certificate",
    "assert_no_family_name_leakage",
    "assert_outcome_hidden",
    "canonical_sha",
    "compile_proposal",
    "proposal_json_schema",
]
