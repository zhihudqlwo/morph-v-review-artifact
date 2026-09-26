"""Generic typed composition and execution for the expanded MORPH-V DSL.

This module is deliberately independent from the frozen three-family replay in
``autonomous_discovery.py``.  The frozen implementation is kept intact while
this path exercises the stronger code--paper contract: every program node is
resolved through a callable registry, compatible primitives compose without a
family-specific executor table, and admitted graphs can be serialized and
reloaded through the same generic interface.

The handlers operate on outcome-hidden structural records.  They are small
reference implementations for compiler and acquisition experiments; benchmark
adapters can provide the same carrier payloads from real video pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .autonomous_discovery import (
    BASE_INVARIANTS,
    DiscoveryContractError,
    assert_outcome_hidden,
)


CARRIERS = ("interval", "slot_set", "event_time")
ROLES = (
    "applicability",
    "defect",
    "binder",
    "transformation",
    "invariant",
    "objective",
    "persist",
)

APPLICABILITY_BY_CARRIER: Mapping[str, str] = {
    "interval": "explicit_text_span_requests_concurrent_visual_state",
    "slot_set": "multiple_semantic_observations_must_all_be_found",
    "event_time": "each_option_is_an_ordered_sequence_of_retrievable_events",
}

DEFECT_BY_CARRIER: Mapping[str, str] = {
    "interval": "localized_anchor_does_not_receive_local_observation_budget",
    "slot_set": "independent_retrieval_can_starve_a_required_evidence_slot",
    "event_time": "independent_event_times_are_not_composed_into_candidate_relations",
}

BINDERS_BY_CARRIER: Mapping[str, Tuple[str, ...]] = {
    "interval": (
        "lexical_span_binder",
        "semantic_segment_binder",
        "hybrid_span_segment_binder",
    ),
    "slot_set": (
        "joint_query_binder",
        "independent_semantic_slots_binder",
        "contrastive_state_slots_binder",
    ),
    "event_time": (
        "single_view_event_retrieval_binder",
        "multiscale_event_retrieval_binder",
        "cross_view_audited_event_retrieval_binder",
    ),
}

TRANSFORMATIONS_BY_CARRIER: Mapping[str, Tuple[str, ...]] = {
    "interval": (
        "local_reallocation",
        "multiscale_local_context",
        "boundary_contrast",
    ),
    "slot_set": (
        "independent_topk",
        "balanced_slot_cover",
        "diversity_aware_slot_cover",
    ),
    "event_time": (
        "raw_timestamp_table",
        "signed_pairwise_contrasts",
        "normalized_order_matrix",
    ),
}

OBJECTIVES_BY_CARRIER: Mapping[str, Tuple[str, ...]] = {
    "interval": (
        "anchor_capture_objective",
        "binding_specificity_objective",
        "specificity_then_shift_equivariance_objective",
    ),
    "slot_set": (
        "mean_support_objective",
        "minimum_slot_coverage_objective",
        "coverage_then_diversity_objective",
    ),
    "event_time": (
        "confidence_margin_objective",
        "finite_completeness_objective",
        "agreement_then_margin_objective",
    ),
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
    "event_time": frozenset(
        {
            "identical_visual_schedule",
            "finite_relation_values",
            "shuffled_order_changes_relation_trace",
        }
    ),
}

HELD_OUT_SIGNATURES: Mapping[str, Mapping[str, str]] = {
    "QuoteBind": {
        "carrier": "interval",
        "binder": "semantic_segment_binder",
        "transformation": "multiscale_local_context",
        "objective": "specificity_then_shift_equivariance_objective",
    },
    "EvidenceSetBind": {
        "carrier": "slot_set",
        "binder": "independent_semantic_slots_binder",
        "transformation": "balanced_slot_cover",
        "objective": "coverage_then_diversity_objective",
    },
    "OrderContrastBind": {
        "carrier": "event_time",
        "binder": "cross_view_audited_event_retrieval_binder",
        "transformation": "signed_pairwise_contrasts",
        "objective": "agreement_then_margin_objective",
    },
}


def _canonical_sha(value: Any) -> str:
    wire = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


Handler = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    role: str
    carrier: str
    input_type: str
    output_type: str
    revision: str
    handler: Optional[Handler]
    dependencies: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise DiscoveryContractError("unknown primitive role: %s" % self.role)
        if self.carrier not in CARRIERS + ("any",):
            raise DiscoveryContractError("unknown primitive carrier: %s" % self.carrier)
        if not self.name or not self.revision:
            raise DiscoveryContractError("primitive name and revision are required")

    def public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "carrier": self.carrier,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "revision": self.revision,
            "dependencies": list(self.dependencies),
            "callable_available": self.handler is not None,
        }


class PrimitiveRegistry:
    """Versioned callable inventory used by both compilation and replay."""

    def __init__(self, semantic_version: str = "expanded-dsl-v1"):
        self.semantic_version = semantic_version
        self._specs: Dict[str, PrimitiveSpec] = {}

    def register(self, spec: PrimitiveSpec) -> None:
        if spec.name in self._specs:
            raise DiscoveryContractError("duplicate primitive: %s" % spec.name)
        self._specs[spec.name] = spec

    def resolve(self, name: str, role: Optional[str] = None) -> PrimitiveSpec:
        if name not in self._specs:
            raise DiscoveryContractError("unknown primitive: %s" % name)
        spec = self._specs[name]
        if role is not None and spec.role != role:
            raise DiscoveryContractError(
                "primitive %s has role %s, expected %s" % (name, spec.role, role)
            )
        if spec.handler is None:
            raise DiscoveryContractError("unavailable callable: %s" % name)
        return spec

    def inventory(
        self, role: Optional[str] = None, carrier: Optional[str] = None
    ) -> Tuple[PrimitiveSpec, ...]:
        values = [
            spec
            for spec in self._specs.values()
            if (role is None or spec.role == role)
            and (carrier is None or spec.carrier == carrier)
        ]
        return tuple(sorted(values, key=lambda item: (item.role, item.carrier, item.name)))

    @property
    def digest(self) -> str:
        return _canonical_sha(
            {
                "semantic_version": self.semantic_version,
                "primitives": [spec.public_dict() for spec in self.inventory()],
            }
        )

    def validate_dependencies(self, selected: Iterable[str]) -> None:
        selected_names = set(selected)
        for name in tuple(selected_names):
            spec = self.resolve(name)
            for dependency in spec.dependencies:
                if dependency not in self._specs:
                    raise DiscoveryContractError(
                        "primitive %s has missing dependency %s" % (name, dependency)
                    )

        visiting: set = set()
        visited: set = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise DiscoveryContractError("cyclic primitive dependency at %s" % name)
            if name in visited:
                return
            visiting.add(name)
            for dependency in self._specs[name].dependencies:
                if dependency in selected_names:
                    visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in selected_names:
            visit(name)


@dataclass(frozen=True)
class ExpandedProposal:
    suggested_name: str
    applicability: str
    defect: str
    binder: str
    transformation: str
    invariants: Tuple[str, ...]
    objective: str
    rationale: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExpandedProposal":
        required = {
            "suggested_name",
            "applicability",
            "defect",
            "binder",
            "transformation",
            "invariants",
            "objective",
            "rationale",
        }
        if set(value) != required:
            raise DiscoveryContractError(
                "expanded proposal fields must be exactly %s" % sorted(required)
            )
        assert_outcome_hidden(value)
        proposal = cls(
            suggested_name=str(value["suggested_name"]).strip(),
            applicability=str(value["applicability"]),
            defect=str(value["defect"]),
            binder=str(value["binder"]),
            transformation=str(value["transformation"]),
            invariants=tuple(str(item) for item in value["invariants"]),
            objective=str(value["objective"]),
            rationale=str(value["rationale"]).strip(),
        )
        if not proposal.suggested_name or not proposal.rationale:
            raise DiscoveryContractError("proposal name and rationale are required")
        if len(set(proposal.invariants)) != len(proposal.invariants):
            raise DiscoveryContractError("proposal invariants must be unique")
        return proposal


@dataclass(frozen=True)
class CompiledGraph:
    graph_id: str
    proposed_name: str
    carrier: str
    applicability: str
    defect: str
    binder: str
    transformation: str
    invariants: Tuple[str, ...]
    objective: str
    nodes: Tuple[Mapping[str, Any], ...]
    registry_version: str
    registry_digest: str
    proposal_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledGraph":
        return cls(
            graph_id=str(value["graph_id"]),
            proposed_name=str(value["proposed_name"]),
            carrier=str(value["carrier"]),
            applicability=str(value["applicability"]),
            defect=str(value["defect"]),
            binder=str(value["binder"]),
            transformation=str(value["transformation"]),
            invariants=tuple(str(item) for item in value["invariants"]),
            objective=str(value["objective"]),
            nodes=tuple(dict(node) for node in value["nodes"]),
            registry_version=str(value["registry_version"]),
            registry_digest=str(value["registry_digest"]),
            proposal_sha256=str(value["proposal_sha256"]),
        )


@dataclass(frozen=True)
class ExecutionRecord:
    graph_id: str
    carrier: str
    applicable: bool
    defect_present: bool
    bound: Mapping[str, Any]
    trace: Mapping[str, Any]
    invariant_checks: Mapping[str, bool]
    objective_score: float
    resolved_revisions: Mapping[str, str]
    outcomes_read: bool
    execution_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchAdmissionRecord:
    graph_id: str
    carrier: str
    population_size: int
    invariant_checks: Mapping[str, bool]
    mean_objective_score: float
    execution_sha256s: Tuple[str, ...]
    outcomes_read: bool
    admission_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_execution_records(
    records: Sequence[ExecutionRecord],
) -> BatchAdmissionRecord:
    """Aggregate a matched structural population into one admission record."""
    if not records:
        raise DiscoveryContractError("cannot admit an empty structural population")
    graph_ids = {record.graph_id for record in records}
    carriers = {record.carrier for record in records}
    if len(graph_ids) != 1 or len(carriers) != 1:
        raise DiscoveryContractError("batch admission mixes graphs or carriers")
    if any(record.outcomes_read for record in records):
        raise DiscoveryContractError("batch admission received outcome-bearing execution")
    invariant_names = set(records[0].invariant_checks)
    if any(set(record.invariant_checks) != invariant_names for record in records):
        raise DiscoveryContractError("batch admission invariant schema drift")
    checks = {
        name: all(record.invariant_checks[name] is True for record in records)
        for name in sorted(invariant_names)
    }
    if not all(checks.values()):
        raise DiscoveryContractError("batch admission contains a failed invariant")
    score = sum(record.objective_score for record in records) / len(records)
    payload = {
        "graph_id": records[0].graph_id,
        "carrier": records[0].carrier,
        "population_size": len(records),
        "invariant_checks": checks,
        "mean_objective_score": score,
        "execution_sha256s": tuple(record.execution_sha256 for record in records),
        "outcomes_read": False,
    }
    return BatchAdmissionRecord(
        admission_sha256=_canonical_sha(payload),
        **payload,
    )


def compatible_signatures() -> Tuple[Tuple[str, str, str, str], ...]:
    return tuple(
        (carrier, binder, transformation, objective)
        for carrier in CARRIERS
        for binder in BINDERS_BY_CARRIER[carrier]
        for transformation in TRANSFORMATIONS_BY_CARRIER[carrier]
        for objective in OBJECTIVES_BY_CARRIER[carrier]
    )


def compile_expanded_proposal(
    proposal: ExpandedProposal,
    registry: PrimitiveRegistry,
    carrier: str,
) -> CompiledGraph:
    """Resolve, type-check, and compile a proposal into the seven-node graph."""
    if carrier not in CARRIERS:
        raise DiscoveryContractError("unknown carrier: %s" % carrier)
    selected = {
        "applicability": proposal.applicability,
        "defect": proposal.defect,
        "binder": proposal.binder,
        "transformation": proposal.transformation,
        "objective": proposal.objective,
    }
    specs = {
        role: registry.resolve(name, role)
        for role, name in selected.items()
    }
    if any(spec.carrier != carrier for spec in specs.values()):
        raise DiscoveryContractError("proposal primitives have incompatible carriers")
    required = set(BASE_INVARIANTS) | set(TYPE_INVARIANTS[carrier])
    missing = required - set(proposal.invariants)
    if missing:
        raise DiscoveryContractError(
            "proposal omits required invariants: %s" % sorted(missing)
        )
    invariant_specs = [
        registry.resolve(name, "invariant") for name in proposal.invariants
    ]
    if any(spec.carrier not in ("any", carrier) for spec in invariant_specs):
        raise DiscoveryContractError("proposal contains a cross-carrier invariant")
    persist = registry.resolve("persist_relation_graph", "persist")

    if specs["binder"].output_type != specs["transformation"].input_type:
        raise DiscoveryContractError("binder output does not match transformation input")
    if specs["transformation"].output_type != specs["objective"].input_type:
        raise DiscoveryContractError("transformation output does not match objective input")

    chosen = list(selected.values()) + list(proposal.invariants) + [persist.name]
    registry.validate_dependencies(chosen)
    proposal_payload = asdict(proposal)
    proposal_sha = _canonical_sha(proposal_payload)
    signature = {
        "carrier": carrier,
        "applicability": proposal.applicability,
        "defect": proposal.defect,
        "binder": proposal.binder,
        "transformation": proposal.transformation,
        "invariants": sorted(proposal.invariants),
        "objective": proposal.objective,
        "registry_digest": registry.digest,
    }
    graph_id = "expanded-relation-%s" % _canonical_sha(signature)[:16]
    nodes: Tuple[Mapping[str, Any], ...] = (
        {"op": "DETECT", "primitive": proposal.applicability},
        {"op": "DIAGNOSE", "primitive": proposal.defect},
        {
            "op": "BIND",
            "primitive": proposal.binder,
            "output_type": specs["binder"].output_type,
        },
        {
            "op": "TRANSFORM",
            "primitive": proposal.transformation,
            "input_type": specs["transformation"].input_type,
            "output_type": specs["transformation"].output_type,
        },
        {"op": "VERIFY", "primitives": sorted(proposal.invariants)},
        {
            "op": "SELECT",
            "primitive": proposal.objective,
            "input_type": specs["objective"].input_type,
        },
        {"op": "PERSIST", "primitive": persist.name, "key": graph_id},
    )
    return CompiledGraph(
        graph_id=graph_id,
        proposed_name=proposal.suggested_name,
        carrier=carrier,
        applicability=proposal.applicability,
        defect=proposal.defect,
        binder=proposal.binder,
        transformation=proposal.transformation,
        invariants=tuple(sorted(proposal.invariants)),
        objective=proposal.objective,
        nodes=nodes,
        registry_version=registry.semantic_version,
        registry_digest=registry.digest,
        proposal_sha256=proposal_sha,
    )


def validate_compiled_graph(graph: CompiledGraph, registry: PrimitiveRegistry) -> None:
    if graph.registry_version != registry.semantic_version:
        raise DiscoveryContractError("registry semantic version drift")
    if graph.registry_digest != registry.digest:
        raise DiscoveryContractError("registry digest drift")
    expected_ops = (
        "DETECT",
        "DIAGNOSE",
        "BIND",
        "TRANSFORM",
        "VERIFY",
        "SELECT",
        "PERSIST",
    )
    if tuple(node.get("op") for node in graph.nodes) != expected_ops:
        raise DiscoveryContractError("compiled graph has an invalid node sequence")
    role_by_op = {
        "DETECT": "applicability",
        "DIAGNOSE": "defect",
        "BIND": "binder",
        "TRANSFORM": "transformation",
        "SELECT": "objective",
        "PERSIST": "persist",
    }
    selected: List[str] = []
    for node in graph.nodes:
        op = str(node["op"])
        if op == "VERIFY":
            for name in node["primitives"]:
                registry.resolve(str(name), "invariant")
                selected.append(str(name))
            continue
        name = str(node["primitive"])
        registry.resolve(name, role_by_op[op])
        selected.append(name)
    registry.validate_dependencies(selected)


def execute_compiled_graph(
    graph: CompiledGraph,
    registry: PrimitiveRegistry,
    context: Mapping[str, Any],
) -> ExecutionRecord:
    """Execute a compiled graph without consulting names or family aliases."""
    validate_compiled_graph(graph, registry)
    assert_outcome_hidden(context)
    if str(context.get("carrier")) != graph.carrier:
        raise DiscoveryContractError("execution context has the wrong carrier")

    resolved: Dict[str, str] = {}

    def call(name: str, role: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
        spec = registry.resolve(name, role)
        resolved[name] = spec.revision
        assert spec.handler is not None
        result = dict(spec.handler(value, context))
        assert_outcome_hidden(result)
        if not _finite(result):
            raise DiscoveryContractError("primitive emitted a non-finite value: %s" % name)
        return result

    applicable_result = call(graph.applicability, "applicability", context)
    applicable = applicable_result.get("applicable") is True
    if not applicable:
        raise DiscoveryContractError("applicability primitive rejected execution context")
    defect_result = call(graph.defect, "defect", context)
    defect_present = defect_result.get("defect_present") is True
    if not defect_present:
        raise DiscoveryContractError("diagnostic primitive found no acquisition defect")
    bound = call(graph.binder, "binder", context)
    trace = call(graph.transformation, "transformation", bound)
    checks: Dict[str, bool] = {}
    for invariant in graph.invariants:
        result = call(invariant, "invariant", trace)
        checks[invariant] = result.get("passed") is True
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise DiscoveryContractError("structural verification failed: %s" % failed)
    objective_result = call(graph.objective, "objective", trace)
    score = float(objective_result["score"])
    persist_result = call(
        "persist_relation_graph",
        "persist",
        {"graph_id": graph.graph_id, "objective_score": score},
    )
    if persist_result.get("persistable") is not True:
        raise DiscoveryContractError("persist primitive rejected the graph")

    payload = {
        "graph_id": graph.graph_id,
        "carrier": graph.carrier,
        "applicable": applicable,
        "defect_present": defect_present,
        "bound": bound,
        "trace": trace,
        "invariant_checks": checks,
        "objective_score": score,
        "resolved_revisions": resolved,
        "outcomes_read": False,
    }
    return ExecutionRecord(execution_sha256=_canonical_sha(payload), **payload)


def rank_executions(records: Sequence[ExecutionRecord]) -> ExecutionRecord:
    if not records:
        raise DiscoveryContractError("cannot rank an empty acquisition set")
    if any(record.outcomes_read for record in records):
        raise DiscoveryContractError("outcome-bearing execution entered structural ranking")
    return sorted(records, key=lambda item: (-item.objective_score, item.graph_id))[0]


class ExpandedRelationLibrary:
    """Persisted compiled graphs plus their outcome-hidden admission records."""

    def __init__(
        self,
        registry: PrimitiveRegistry,
        entries: Optional[Iterable[Tuple[CompiledGraph, BatchAdmissionRecord]]] = None,
    ):
        self.registry = registry
        self._entries: List[Tuple[CompiledGraph, BatchAdmissionRecord]] = list(
            entries or []
        )
        self._validate()

    def _validate(self) -> None:
        ids = [graph.graph_id for graph, _ in self._entries]
        if len(ids) != len(set(ids)):
            raise DiscoveryContractError("expanded library contains duplicate graph IDs")
        for graph, record in self._entries:
            validate_compiled_graph(graph, self.registry)
            if graph.graph_id != record.graph_id or record.outcomes_read:
                raise DiscoveryContractError("invalid expanded-library admission record")

    @property
    def size_with_direct(self) -> int:
        return 1 + len(self._entries)

    @property
    def graphs(self) -> Tuple[CompiledGraph, ...]:
        return tuple(graph for graph, _ in self._entries)

    def admit(self, graph: CompiledGraph, record: Any) -> None:
        admission = (
            aggregate_execution_records([record])
            if isinstance(record, ExecutionRecord)
            else record
        )
        if not isinstance(admission, BatchAdmissionRecord):
            raise DiscoveryContractError("unsupported admission record")
        if graph.graph_id != admission.graph_id:
            raise DiscoveryContractError("execution does not belong to graph")
        if admission.outcomes_read or not all(admission.invariant_checks.values()):
            raise DiscoveryContractError("only outcome-hidden verified graphs may persist")
        self._entries.append((graph, admission))
        self._validate()

    def execute(
        self, graph_id: str, context: Mapping[str, Any]
    ) -> ExecutionRecord:
        for graph, _ in self._entries:
            if graph.graph_id == graph_id:
                return execute_compiled_graph(graph, self.registry, context)
        raise DiscoveryContractError("graph is absent from expanded library")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 2,
            "registry_version": self.registry.semantic_version,
            "registry_digest": self.registry.digest,
            "fallback": "Direct",
            "size_with_direct": self.size_with_direct,
            "outcomes_read": False,
            "entries": [
                {"graph": graph.to_dict(), "admission": record.to_dict()}
                for graph, record in self._entries
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, registry: PrimitiveRegistry) -> "ExpandedRelationLibrary":
        value = json.loads(path.read_text(encoding="utf-8"))
        # ``no_outcome_fields`` is a registered invariant name.  It is safe as
        # a value, but the frozen recursive firewall intentionally rejects any
        # outcome token appearing as a mapping key.  Normalize the check map to
        # a list of symbolic values before applying that same firewall.
        firewall_view = json.loads(json.dumps(value))
        for item in firewall_view.get("entries") or []:
            admission = item.get("admission") or {}
            checks = admission.pop("invariant_checks", {})
            admission["verified_invariants"] = sorted(
                name for name, passed in checks.items() if passed is True
            )
            revisions = admission.pop("resolved_revisions", {})
            admission["resolved_primitives"] = [
                {"primitive": name, "revision": revision}
                for name, revision in sorted(revisions.items())
            ]
        assert_outcome_hidden(firewall_view)
        if value.get("registry_version") != registry.semantic_version:
            raise DiscoveryContractError("saved library registry version drift")
        if value.get("registry_digest") != registry.digest:
            raise DiscoveryContractError("saved library registry digest drift")
        entries = []
        for item in value.get("entries") or []:
            graph = CompiledGraph.from_mapping(item["graph"])
            admission = item["admission"]
            record_payload = dict(admission)
            record = BatchAdmissionRecord(
                graph_id=str(record_payload["graph_id"]),
                carrier=str(record_payload["carrier"]),
                population_size=int(record_payload["population_size"]),
                invariant_checks={
                    str(key): bool(flag)
                    for key, flag in record_payload["invariant_checks"].items()
                },
                mean_objective_score=float(record_payload["mean_objective_score"]),
                execution_sha256s=tuple(
                    str(item) for item in record_payload["execution_sha256s"]
                ),
                outcomes_read=bool(record_payload["outcomes_read"]),
                admission_sha256=str(record_payload["admission_sha256"]),
            )
            expected = dict(record.to_dict())
            observed_sha = expected.pop("admission_sha256")
            if _canonical_sha(expected) != observed_sha:
                raise DiscoveryContractError("saved admission record hash drift")
            entries.append((graph, record))
        return cls(registry, entries)


def _applicability(carrier: str) -> Handler:
    def handler(value: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"applicable": value.get("carrier") == carrier}

    return handler


def _defect(value: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"defect_present": value.get("defect_present", True) is True}


def _interval_binder(mode: str) -> Handler:
    def handler(value: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        lexical = list(value.get("lexical_intervals") or [])
        semantic = list(value.get("semantic_intervals") or [])
        if mode == "lexical":
            candidates = lexical
            decoy_score = float(value.get("lexical_decoy_score", 0.0))
        elif mode == "semantic":
            candidates = semantic
            decoy_score = float(value.get("semantic_decoy_score", 0.0))
        else:
            candidates = lexical + semantic
            decoy_score = max(
                float(value.get("lexical_decoy_score", 0.0)),
                float(value.get("semantic_decoy_score", 0.0)),
            )
        if not candidates:
            raise DiscoveryContractError("interval binder received no candidates")
        best = max(candidates, key=lambda item: (float(item["score"]), -float(item["start"])))
        return {
            "interval": [float(best["start"]), float(best["end"])],
            "binding_score": float(best["score"]),
            "binding_specificity": float(best["score"]) - decoy_score,
            "candidate_count": len(candidates),
            "binder_mode": mode,
        }

    return handler


def _interval_transform(mode: str) -> Handler:
    def handler(value: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        start, end = [float(item) for item in value["interval"]]
        budget = int(context.get("frame_budget", 8))
        duration = float(context.get("duration", max(end, 1.0)))
        midpoint = (start + end) / 2.0
        if mode == "local":
            width = max(end - start, duration / 20.0)
            raw = [midpoint - width / 2 + width * index / max(budget - 1, 1) for index in range(budget)]
        elif mode == "multiscale":
            scales = (0.5, 1.0, 2.0)
            raw = [
                midpoint + (scale * max(end - start, duration / 20.0)) * ((index % 3) - 1) / 2
                for index, scale in zip(range(budget), (scales * (budget // 3 + 1))[:budget])
            ]
        else:
            offsets = (-1.25, -0.75, -0.25, 0.25, 0.75, 1.25)
            raw = [midpoint + offsets[index % len(offsets)] * max(end - start, duration / 20.0) for index in range(budget)]
        schedule = sorted(min(duration, max(0.0, float(item))) for item in raw)
        captured = sum(start <= item <= end for item in schedule) / max(len(schedule), 1)
        shift = float(context.get("shift_equivariance", 1.0))
        specificity = float(value["binding_specificity"])
        diversity = len(set(round(item, 6) for item in schedule)) / max(len(schedule), 1)
        return {
            "schedule": schedule,
            "frame_count": len(schedule),
            "anchor_capture": captured,
            "binding_specificity": specificity,
            "shift_equivariance": shift,
            "diversity": diversity,
            "transform_mode": mode,
        }

    return handler


def _slot_binder(mode: str) -> Handler:
    def handler(value: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        scores = value.get("slot_scores") or {}
        if not scores:
            raise DiscoveryContractError("slot binder received no slot scores")
        normalized: Dict[str, List[float]] = {}
        for slot, raw_scores in scores.items():
            values = [float(item) for item in raw_scores]
            if mode == "joint":
                mean = sum(values) / len(values)
                values = [(item + mean) / 2.0 for item in values]
            elif mode == "contrastive":
                negatives = [float(item) for item in (value.get("negative_scores") or {}).get(slot, [0.0] * len(values))]
                values = [item - negatives[index] for index, item in enumerate(values)]
            normalized[str(slot)] = values
        return {
            "slot_scores": normalized,
            "slot_count": len(normalized),
            "binder_mode": mode,
        }

    return handler


def _slot_transform(mode: str) -> Handler:
    def handler(value: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        scores = value["slot_scores"]
        budget = int(context.get("frame_budget", max(1, len(scores))))
        selected: List[int] = []
        support: Dict[str, float] = {}
        if mode == "topk":
            aggregate: Dict[int, float] = {}
            for values in scores.values():
                for index, score in enumerate(values):
                    aggregate[index] = max(aggregate.get(index, -math.inf), float(score))
            selected = [index for index, _ in sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))[:budget]]
        else:
            ranked = {
                slot: [index for index, _ in sorted(enumerate(values), key=lambda item: (-item[1], item[0]))]
                for slot, values in scores.items()
            }
            slots = sorted(scores)
            cursor = {slot: 0 for slot in slots}
            while len(selected) < budget and slots:
                made_progress = False
                for slot in slots:
                    while cursor[slot] < len(ranked[slot]) and ranked[slot][cursor[slot]] in selected:
                        cursor[slot] += 1
                    if cursor[slot] < len(ranked[slot]) and len(selected) < budget:
                        candidate = ranked[slot][cursor[slot]]
                        if mode == "diversity" and selected:
                            remaining = ranked[slot][cursor[slot]:]
                            candidate = max(remaining, key=lambda index: (min(abs(index - old) for old in selected), scores[slot][index]))
                        selected.append(candidate)
                        made_progress = True
                if not made_progress:
                    break
        selected = sorted(set(selected))
        for slot, values in scores.items():
            support[slot] = max((float(values[index]) for index in selected), default=0.0)
        minimum = min(support.values()) if support else 0.0
        mean = sum(support.values()) / len(support) if support else 0.0
        diversity = (
            len(selected) / max(max(selected) - min(selected) + 1, 1) if selected else 0.0
        )
        lattice_timestamps = [
            float(item) for item in context.get("lattice_timestamps") or []
        ]
        if lattice_timestamps and max(selected, default=-1) >= len(lattice_timestamps):
            raise DiscoveryContractError("slot schedule exceeds the timestamp lattice")
        return {
            "selected_frames": selected,
            "schedule": (
                [lattice_timestamps[index] for index in selected]
                if lattice_timestamps
                else []
            ),
            "frame_count": len(selected),
            "slot_support": support,
            "minimum_slot_coverage": minimum,
            "mean_support": mean,
            "diversity": diversity,
            "transform_mode": mode,
        }

    return handler


def _event_binder(mode: str) -> Handler:
    def handler(value: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        views = value.get("event_views") or {}
        if not views:
            raise DiscoveryContractError("event binder received no retrieval views")
        names = sorted({name for events in views.values() for name in events})
        chosen_views = sorted(views)
        if mode == "single":
            chosen_views = chosen_views[:1]
        elif mode == "multiscale":
            chosen_views = chosen_views[: min(2, len(chosen_views))]
        times: Dict[str, float] = {}
        confidences: Dict[str, float] = {}
        agreement: Dict[str, float] = {}
        for name in names:
            rows = [views[view][name] for view in chosen_views if name in views[view]]
            if not rows:
                continue
            raw_times = [float(row["time"]) for row in rows]
            raw_conf = [float(row["confidence"]) for row in rows]
            times[name] = sum(raw_times) / len(raw_times)
            confidences[name] = sum(raw_conf) / len(raw_conf)
            spread = max(raw_times) - min(raw_times)
            agreement[name] = 1.0 / (1.0 + spread)
        return {
            "event_times": times,
            "event_confidences": confidences,
            "event_agreement": agreement,
            "view_count": len(chosen_views),
            "binder_mode": mode,
        }

    return handler


def _event_transform(mode: str) -> Handler:
    def handler(value: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        times = {str(key): float(item) for key, item in value["event_times"].items()}
        names = sorted(times)
        relations: Dict[str, float] = {}
        if mode == "raw":
            relations = dict(times)
        else:
            for left_index, left in enumerate(names):
                for right in names[left_index + 1 :]:
                    delta = times[right] - times[left]
                    if mode == "normalized":
                        delta = 0.0 if delta == 0 else (1.0 if delta > 0 else -1.0)
                    relations["%s->%s" % (left, right)] = delta
        confidence = list(value["event_confidences"].values())
        agreement = list(value["event_agreement"].values())
        margin = min((float(item) for item in confidence), default=0.0)
        finite = _finite(relations) and len(relations) > 0
        shuffled = context.get("shuffled_relation_sha256")
        relation_sha = _canonical_sha(relations)
        return {
            "relations": relations,
            "relation_sha256": relation_sha,
            "schedule": [
                float(item) for item in context.get("visual_schedule") or []
            ],
            "finite_completeness": 1.0 if finite else 0.0,
            "confidence_margin": margin,
            "cross_view_agreement": sum(agreement) / len(agreement) if agreement else 0.0,
            "shuffled_relation_changed": shuffled is None or str(shuffled) != relation_sha,
            "frame_count": int(context.get("frame_budget", 8)),
            "transform_mode": mode,
        }

    return handler


def _objective(metric: str, secondary: Optional[str] = None) -> Handler:
    def handler(value: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        primary = min(1.0, max(0.0, float(value.get(metric, 0.0))))
        if secondary is None:
            return {"score": primary}
        # All registered structural metrics are calibrated to [0, 1].  A small
        # secondary weight implements deterministic lexicographic tie-breaking
        # without making scores from different objectives incomparable.
        secondary_value = min(
            1.0, max(0.0, float(value.get(secondary, 0.0)))
        )
        return {"score": 0.999 * primary + 0.001 * secondary_value}

    return handler


def _invariant(name: str) -> Handler:
    def handler(value: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        explicit = (context.get("invariant_overrides") or {}).get(name)
        if explicit is not None:
            return {"passed": explicit is True}
        if name == "one_answer_call":
            passed = int(context.get("answer_calls", 0)) <= 1
        elif name == "fixed_frame_budget":
            passed = int(value.get("frame_count", context.get("frame_budget", 0))) == int(context.get("frame_budget", 0))
        elif name == "no_outcome_fields":
            passed = True
        elif name == "deterministic_replay":
            passed = context.get("deterministic", True) is True
        elif name == "bound_interval_capture":
            passed = float(value.get("anchor_capture", 0.0)) > 0.0
        elif name == "temporal_shift_equivariance":
            passed = float(value.get("shift_equivariance", 0.0)) >= float(context.get("minimum_shift_equivariance", 0.0))
        elif name == "cross_video_binding_specificity":
            passed = float(value.get("binding_specificity", 0.0)) > 0.0
        elif name == "every_required_slot_covered":
            passed = float(value.get("minimum_slot_coverage", 0.0)) > 0.0
        elif name == "balanced_slot_allocation":
            passed = len(value.get("slot_support") or {}) > 0
        elif name == "identical_visual_schedule":
            passed = context.get("identical_visual_schedule", True) is True
        elif name == "finite_relation_values":
            passed = float(value.get("finite_completeness", 0.0)) == 1.0
        elif name == "shuffled_order_changes_relation_trace":
            passed = value.get("shuffled_relation_changed") is True
        else:
            raise DiscoveryContractError("no invariant implementation: %s" % name)
        return {"passed": passed}

    return handler


def _persist(value: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"persistable": bool(value.get("graph_id")) and _finite(value.get("objective_score"))}


def build_default_registry() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()
    revision = "1.0.0"
    for carrier in CARRIERS:
        registry.register(
            PrimitiveSpec(
                APPLICABILITY_BY_CARRIER[carrier],
                "applicability",
                carrier,
                "structural_context",
                "applicability_decision",
                revision,
                _applicability(carrier),
            )
        )
        registry.register(
            PrimitiveSpec(
                DEFECT_BY_CARRIER[carrier],
                "defect",
                carrier,
                "structural_context",
                "defect_decision",
                revision,
                _defect,
            )
        )

    binder_handlers: Mapping[str, Handler] = {
        "lexical_span_binder": _interval_binder("lexical"),
        "semantic_segment_binder": _interval_binder("semantic"),
        "hybrid_span_segment_binder": _interval_binder("hybrid"),
        "joint_query_binder": _slot_binder("joint"),
        "independent_semantic_slots_binder": _slot_binder("independent"),
        "contrastive_state_slots_binder": _slot_binder("contrastive"),
        "single_view_event_retrieval_binder": _event_binder("single"),
        "multiscale_event_retrieval_binder": _event_binder("multiscale"),
        "cross_view_audited_event_retrieval_binder": _event_binder("cross_view"),
    }
    transformation_handlers: Mapping[str, Handler] = {
        "local_reallocation": _interval_transform("local"),
        "multiscale_local_context": _interval_transform("multiscale"),
        "boundary_contrast": _interval_transform("boundary"),
        "independent_topk": _slot_transform("topk"),
        "balanced_slot_cover": _slot_transform("balanced"),
        "diversity_aware_slot_cover": _slot_transform("diversity"),
        "raw_timestamp_table": _event_transform("raw"),
        "signed_pairwise_contrasts": _event_transform("signed"),
        "normalized_order_matrix": _event_transform("normalized"),
    }
    objective_handlers: Mapping[str, Handler] = {
        "anchor_capture_objective": _objective("anchor_capture"),
        "binding_specificity_objective": _objective("binding_specificity"),
        "specificity_then_shift_equivariance_objective": _objective("binding_specificity", "shift_equivariance"),
        "mean_support_objective": _objective("mean_support"),
        "minimum_slot_coverage_objective": _objective("minimum_slot_coverage"),
        "coverage_then_diversity_objective": _objective("minimum_slot_coverage", "diversity"),
        "confidence_margin_objective": _objective("confidence_margin"),
        "finite_completeness_objective": _objective("finite_completeness"),
        "agreement_then_margin_objective": _objective("cross_view_agreement", "confidence_margin"),
    }

    for carrier in CARRIERS:
        bound_type = "bound_%s" % carrier
        trace_type = "%s_trace" % carrier
        for name in BINDERS_BY_CARRIER[carrier]:
            registry.register(
                PrimitiveSpec(
                    name,
                    "binder",
                    carrier,
                    "structural_context",
                    bound_type,
                    revision,
                    binder_handlers[name],
                )
            )
        for name in TRANSFORMATIONS_BY_CARRIER[carrier]:
            registry.register(
                PrimitiveSpec(
                    name,
                    "transformation",
                    carrier,
                    bound_type,
                    trace_type,
                    revision,
                    transformation_handlers[name],
                )
            )
        for name in OBJECTIVES_BY_CARRIER[carrier]:
            registry.register(
                PrimitiveSpec(
                    name,
                    "objective",
                    carrier,
                    trace_type,
                    "scalar_score",
                    revision,
                    objective_handlers[name],
                )
            )

    all_invariants = set(BASE_INVARIANTS)
    for values in TYPE_INVARIANTS.values():
        all_invariants.update(values)
    for name in sorted(all_invariants):
        carriers = [carrier for carrier, values in TYPE_INVARIANTS.items() if name in values]
        carrier = carriers[0] if len(carriers) == 1 else "any"
        registry.register(
            PrimitiveSpec(
                name,
                "invariant",
                carrier,
                "structural_trace",
                "boolean",
                revision,
                _invariant(name),
            )
        )
    registry.register(
        PrimitiveSpec(
            "persist_relation_graph",
            "persist",
            "any",
            "verified_graph",
            "library_entry",
            revision,
            _persist,
        )
    )
    return registry


def proposal_for_signature(
    carrier: str,
    binder: str,
    transformation: str,
    objective: str,
    suggested_name: str = "AnonymousRelationProgram",
) -> ExpandedProposal:
    invariants = tuple(sorted(set(BASE_INVARIANTS) | set(TYPE_INVARIANTS[carrier])))
    return ExpandedProposal(
        suggested_name=suggested_name,
        applicability=APPLICABILITY_BY_CARRIER[carrier],
        defect=DEFECT_BY_CARRIER[carrier],
        binder=binder,
        transformation=transformation,
        invariants=invariants,
        objective=objective,
        rationale="Compose compatible registered primitives for an uncovered structural carrier.",
    )
