from __future__ import annotations

from dataclasses import replace

import pytest

from evogroundtest.morph_v.autonomous_discovery import DiscoveryContractError
from evogroundtest.morph_v.expanded_dsl import (
    BINDERS_BY_CARRIER,
    CARRIERS,
    HELD_OUT_SIGNATURES,
    OBJECTIVES_BY_CARRIER,
    TRANSFORMATIONS_BY_CARRIER,
    ExpandedRelationLibrary,
    PrimitiveRegistry,
    PrimitiveSpec,
    build_default_registry,
    compatible_signatures,
    compile_expanded_proposal,
    execute_compiled_graph,
    proposal_for_signature,
)


def context_for(carrier: str):
    if carrier == "interval":
        return {
            "carrier": "interval",
            "defect_present": True,
            "duration": 10.0,
            "frame_budget": 6,
            "lexical_intervals": [
                {"start": 3.5, "end": 5.5, "score": 0.71},
            ],
            "semantic_intervals": [
                {"start": 4.0, "end": 6.0, "score": 0.93},
                {"start": 0.0, "end": 1.0, "score": 0.22},
            ],
            "shift_equivariance": 0.91,
            "answer_calls": 0,
            "deterministic": True,
        }
    if carrier == "slot_set":
        return {
            "carrier": "slot_set",
            "defect_present": True,
            "frame_budget": 3,
            "slot_scores": {
                "state_a": [0.90, 0.10, 0.20, 0.30],
                "state_b": [0.20, 0.86, 0.10, 0.30],
                "state_c": [0.10, 0.20, 0.88, 0.30],
            },
            "negative_scores": {
                "state_a": [0.10, 0.10, 0.10, 0.10],
                "state_b": [0.10, 0.10, 0.10, 0.10],
                "state_c": [0.10, 0.10, 0.10, 0.10],
            },
            "answer_calls": 0,
            "deterministic": True,
        }
    if carrier == "event_time":
        return {
            "carrier": "event_time",
            "defect_present": True,
            "frame_budget": 8,
            "event_views": {
                "coarse": {
                    "arrive": {"time": 1.0, "confidence": 0.82},
                    "open": {"time": 4.0, "confidence": 0.87},
                    "leave": {"time": 7.0, "confidence": 0.84},
                },
                "fine": {
                    "arrive": {"time": 1.1, "confidence": 0.88},
                    "open": {"time": 4.2, "confidence": 0.91},
                    "leave": {"time": 6.9, "confidence": 0.86},
                },
                "audit": {
                    "arrive": {"time": 0.9, "confidence": 0.80},
                    "open": {"time": 4.1, "confidence": 0.89},
                    "leave": {"time": 7.1, "confidence": 0.83},
                },
            },
            "shuffled_relation_sha256": "0" * 64,
            "identical_visual_schedule": True,
            "answer_calls": 0,
            "deterministic": True,
        }
    raise AssertionError(carrier)


def compile_signature(registry, signature, name="AnonymousProgram"):
    proposal = proposal_for_signature(
        carrier=signature["carrier"],
        binder=signature["binder"],
        transformation=signature["transformation"],
        objective=signature["objective"],
        suggested_name=name,
    )
    return compile_expanded_proposal(proposal, registry, signature["carrier"])


def test_inventory_exposes_exactly_81_compatible_signatures() -> None:
    registry = build_default_registry()
    signatures = compatible_signatures()
    assert len(signatures) == 81
    assert len(set(signatures)) == 81
    for carrier in CARRIERS:
        assert len(BINDERS_BY_CARRIER[carrier]) == 3
        assert len(TRANSFORMATIONS_BY_CARRIER[carrier]) == 3
        assert len(OBJECTIVES_BY_CARRIER[carrier]) == 3
        assert sum(item[0] == carrier for item in signatures) == 27
        assert len(registry.inventory("binder", carrier)) == 3
        assert len(registry.inventory("transformation", carrier)) == 3
        assert len(registry.inventory("objective", carrier)) == 3


def test_all_81_signatures_compile_and_execute_through_one_generic_path() -> None:
    registry = build_default_registry()
    for carrier, binder, transformation, objective in compatible_signatures():
        proposal = proposal_for_signature(carrier, binder, transformation, objective)
        graph = compile_expanded_proposal(proposal, registry, carrier)
        assert tuple(node["op"] for node in graph.nodes) == (
            "DETECT",
            "DIAGNOSE",
            "BIND",
            "TRANSFORM",
            "VERIFY",
            "SELECT",
            "PERSIST",
        )
        record = execute_compiled_graph(graph, registry, context_for(carrier))
        assert record.graph_id == graph.graph_id
        assert record.outcomes_read is False
        assert all(record.invariant_checks.values())


@pytest.mark.parametrize("alias", sorted(HELD_OUT_SIGNATURES))
def test_held_out_complete_signatures_execute_without_alias_dispatch(alias: str) -> None:
    registry = build_default_registry()
    signature = HELD_OUT_SIGNATURES[alias]
    graph = compile_signature(registry, signature, name="Neutral-%s" % signature["carrier"])
    record = execute_compiled_graph(graph, registry, context_for(signature["carrier"]))
    assert record.outcomes_read is False
    assert all(record.invariant_checks.values())
    assert record.objective_score > 0.0
    assert alias.lower() not in str(graph.to_dict()).lower()


def test_program_identity_does_not_depend_on_generated_name() -> None:
    registry = build_default_registry()
    signature = HELD_OUT_SIGNATURES["QuoteBind"]
    first = compile_signature(registry, signature, name="FirstNeutralName")
    second = compile_signature(registry, signature, name="SecondNeutralName")
    assert first.graph_id == second.graph_id
    assert first.proposal_sha256 != second.proposal_sha256


def test_cross_carrier_and_unavailable_callables_fail_closed() -> None:
    registry = build_default_registry()
    signature = HELD_OUT_SIGNATURES["QuoteBind"]
    proposal = proposal_for_signature(
        carrier="interval",
        binder="semantic_segment_binder",
        transformation="balanced_slot_cover",
        objective="specificity_then_shift_equivariance_objective",
    )
    with pytest.raises(DiscoveryContractError, match="incompatible carriers"):
        compile_expanded_proposal(proposal, registry, "interval")

    unavailable = PrimitiveRegistry("unavailable-test")
    unavailable.register(
        PrimitiveSpec(
            "missing_binder",
            "binder",
            "interval",
            "structural_context",
            "bound_interval",
            "1.0.0",
            None,
        )
    )
    with pytest.raises(DiscoveryContractError, match="unavailable callable"):
        unavailable.resolve("missing_binder", "binder")


def test_missing_and_cyclic_dependencies_fail_closed() -> None:
    def passthrough(value, context):
        return dict(value)

    missing = PrimitiveRegistry("dependency-test")
    missing.register(
        PrimitiveSpec(
            "a",
            "binder",
            "interval",
            "x",
            "y",
            "1.0.0",
            passthrough,
            dependencies=("absent",),
        )
    )
    with pytest.raises(DiscoveryContractError, match="missing dependency"):
        missing.validate_dependencies(["a"])

    cyclic = PrimitiveRegistry("cycle-test")
    cyclic.register(
        PrimitiveSpec("a", "binder", "interval", "x", "y", "1.0.0", passthrough, dependencies=("b",))
    )
    cyclic.register(
        PrimitiveSpec("b", "binder", "interval", "x", "y", "1.0.0", passthrough, dependencies=("a",))
    )
    with pytest.raises(DiscoveryContractError, match="cyclic"):
        cyclic.validate_dependencies(["a", "b"])


@pytest.mark.parametrize("alias", sorted(HELD_OUT_SIGNATURES))
def test_persisted_held_out_graph_reloads_and_reexecutes(alias: str, tmp_path) -> None:
    registry = build_default_registry()
    signature = HELD_OUT_SIGNATURES[alias]
    context = context_for(signature["carrier"])
    graph = compile_signature(registry, signature)
    initial = execute_compiled_graph(graph, registry, context)

    library = ExpandedRelationLibrary(registry)
    library.admit(graph, initial)
    path = tmp_path / "relation_library.json"
    library.save(path)

    restored = ExpandedRelationLibrary.load(path, registry)
    replay = restored.execute(graph.graph_id, context)
    assert restored.size_with_direct == 2
    assert replay.execution_sha256 == initial.execution_sha256
    assert replay.bound == initial.bound
    assert replay.trace == initial.trace


def test_registry_revision_drift_invalidates_saved_graph(tmp_path) -> None:
    registry = build_default_registry()
    signature = HELD_OUT_SIGNATURES["OrderContrastBind"]
    context = context_for("event_time")
    graph = compile_signature(registry, signature)
    record = execute_compiled_graph(graph, registry, context)
    library = ExpandedRelationLibrary(registry)
    library.admit(graph, record)
    path = tmp_path / "relation_library.json"
    library.save(path)

    drifted = build_default_registry()
    changed = drifted._specs[signature["binder"]]
    drifted._specs[signature["binder"]] = replace(changed, revision="1.0.1")
    with pytest.raises(DiscoveryContractError, match="registry digest drift"):
        ExpandedRelationLibrary.load(path, drifted)
