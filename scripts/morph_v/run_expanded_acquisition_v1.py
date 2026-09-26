#!/usr/bin/env python3
"""Run the expanded MORPH-V acquisition experiment and matched controls.

The default invocation is a no-network preflight.  ``--execute`` is required
before any proposer request is sent.  Answer-model evaluation is intentionally
separate and only consumes frozen L0--L3 libraries produced here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Mapping
import urllib.error
import urllib.request

import yaml

from evogroundtest.morph_v.autonomous_discovery import DiscoveryContractError
from evogroundtest.morph_v.expanded_acquisition import (
    PROCEDURES,
    AcquisitionLimits,
    ProposalEnvelope,
    expanded_proposal_json_schema,
    load_context_bank,
    run_acquisition,
)
from evogroundtest.morph_v.expanded_dsl import build_default_registry


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/morph_v/expanded_acquisition_v1.yaml"


def canonical_sha(value: Any) -> str:
    wire = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _content_from_response(raw: bytes) -> tuple[str, Mapping[str, Any]]:
    text = raw.decode("utf-8", errors="replace")
    content = []
    usage: Mapping[str, Any] = {}
    saw_sse = False
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        saw_sse = True
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        event = json.loads(payload)
        if event.get("error"):
            error = event["error"]
            message = (
                str(error.get("message") or error.get("type") or "endpoint error")
                if isinstance(error, Mapping)
                else "endpoint error"
            )
            raise RuntimeError("proposer endpoint error: %s" % message[:500])
        usage = event.get("usage") or usage
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(str(delta["content"]))
    if saw_sse:
        return "".join(content), usage
    event = json.loads(text)
    if event.get("error"):
        error = event["error"]
        message = (
            str(error.get("message") or error.get("type") or "endpoint error")
            if isinstance(error, Mapping)
            else "endpoint error"
        )
        raise RuntimeError("proposer endpoint error: %s" % message[:500])
    usage = event.get("usage") or {}
    for choice in event.get("choices") or []:
        message = choice.get("message") or {}
        if message.get("content"):
            content.append(str(message["content"]))
    return "".join(content), usage


class HTTPProposer:
    def __init__(self, config: Mapping[str, Any]):
        self.model = str(config["model"])
        endpoint_env = str(config["endpoint_env"])
        api_key_env = str(config["api_key_env"])
        self.endpoint = os.environ.get(
            endpoint_env, str(config.get("endpoint") or "")
        ).strip()
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.endpoint:
            raise DiscoveryContractError(
                "missing proposer endpoint environment variable: %s" % endpoint_env
            )
        if bool(config.get("api_key_required", True)) and not self.api_key:
            raise DiscoveryContractError(
                "missing proposer API-key environment variable: %s" % api_key_env
            )
        self.timeout_seconds = int(config.get("timeout_seconds", 180))
        self.maximum_tokens = int(config.get("maximum_tokens", 1024))
        self.seed = int(config.get("seed", 20260916))
        self.enable_thinking = bool(config.get("enable_thinking", False))
        self.top_k = int(config.get("top_k", 20))
        self.stream = bool(config.get("stream", False))

    def __call__(self, packet: Mapping[str, Any]) -> ProposalEnvelope:
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Compose one typed executable observation program using only the "
                        "provided anonymous structure and primitive registry. Do not use "
                        "benchmark identity or outcomes. Return exactly the JSON schema."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(packet, ensure_ascii=False, sort_keys=True),
                },
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": self.top_k,
            "seed": self.seed,
            "max_tokens": self.maximum_tokens,
            "stream": self.stream,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "morph_v_expanded_proposal_v1",
                    "strict": True,
                    "schema": expanded_proposal_json_schema(),
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                request, timeout=self.timeout_seconds
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError("proposer endpoint returned HTTP_%d" % exc.code) from exc
        latency = time.monotonic() - started
        content, usage = _content_from_response(raw)
        if not content:
            raise RuntimeError("proposer endpoint returned empty content")
        try:
            proposal = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("proposer endpoint returned non-JSON content") from exc
        return ProposalEnvelope(
            proposal=proposal,
            transport={
                "model": self.model,
                "latency_seconds": latency,
                "usage": dict(usage),
                "request_sha256": canonical_sha(body),
                "endpoint_sha256": hashlib.sha256(self.endpoint.encode()).hexdigest(),
                "source": "remote_proposer",
                "enable_thinking": self.enable_thinking,
            },
        )


class ResponseLedgerProposer:
    """Replay raw model responses through the same compiler/executor path."""

    def __init__(self, path: Path, model: str):
        self.path = path.resolve()
        self.model = model
        self.responses = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.position = 0
        if not self.responses:
            raise DiscoveryContractError("empty model-response ledger")

    def __call__(self, packet: Mapping[str, Any]) -> ProposalEnvelope:
        if self.position >= len(self.responses):
            raise DiscoveryContractError("model-response ledger is exhausted")
        proposal = self.responses[self.position]
        self.position += 1
        return ProposalEnvelope(
            proposal=proposal,
            transport={
                "model": self.model,
                "source": "frozen_model_response_ledger",
                "ledger_path": str(self.path),
                "ledger_position": self.position,
                "request_sha256": canonical_sha(packet),
                "response_sha256": canonical_sha(proposal),
                "usage": {},
            },
        )

    def assert_exhausted(self) -> None:
        if self.position != len(self.responses):
            raise DiscoveryContractError(
                "model-response ledger has %d unused rows"
                % (len(self.responses) - self.position)
            )


def _config(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise DiscoveryContractError("invalid expanded acquisition config")
    return value


def preflight(config_path: Path) -> Dict[str, Any]:
    config = _config(config_path)
    context_path = (REPO / str(config["context_bank"])).resolve()
    registry = build_default_registry()
    return {
        "status": "READY" if context_path.is_file() else "WAITING_FOR_CONTEXT_BANK",
        "config": str(config_path),
        "context_bank": str(context_path),
        "context_bank_exists": context_path.is_file(),
        "endpoint_configured": bool(
            os.environ.get(str(config["proposer"]["endpoint_env"]), "").strip()
            or str(config["proposer"].get("endpoint") or "").strip()
        ),
        "api_key_configured": bool(
            os.environ.get(str(config["proposer"]["api_key_env"]), "").strip()
        ),
        "api_key_required": bool(config["proposer"].get("api_key_required", True)),
        "enable_thinking": bool(config["proposer"].get("enable_thinking", False)),
        "registry_version": registry.semantic_version,
        "registry_digest": registry.digest,
        "compatible_signature_count": 81,
        "network_calls": 0,
    }


def run(
    config_path: Path,
    procedure: str,
    proposer_override: Any = None,
) -> Dict[str, Any]:
    config = _config(config_path)
    context_path = (REPO / str(config["context_bank"])).resolve()
    if not context_path.is_file():
        raise FileNotFoundError("expanded carrier context bank is missing: %s" % context_path)
    clusters = load_context_bank(context_path)
    limits = AcquisitionLimits(**dict(config["limits"]))
    output_root = (REPO / str(config["output_root"])).resolve()
    procedures: Iterable[str] = PROCEDURES if procedure == "all" else (procedure,)
    proposer = None
    results = {}
    for name in procedures:
        if name in {
            "morph_v",
            "without_structural_ranking",
            "without_persistence_feedback",
        }:
            if proposer is None:
                proposer = proposer_override or HTTPProposer(config["proposer"])
            current_proposer = proposer
        else:
            current_proposer = None
        results[name] = run_acquisition(
            name,
            clusters,
            build_default_registry(),
            limits,
            proposer=current_proposer,
            output_dir=output_root / name,
            carrier_order=config["protocol"]["carrier_order"],
        )
    if proposer_override is not None and hasattr(proposer_override, "assert_exhausted"):
        proposer_override.assert_exhausted()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--procedure", choices=("all",) + PROCEDURES, default="all")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send proposer requests and write acquisition artifacts",
    )
    parser.add_argument(
        "--response-ledger",
        type=Path,
        help="replay newline-delimited raw model JSON responses instead of calling an endpoint",
    )
    parser.add_argument("--ledger-model", default="unreported-model")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not args.execute:
        print(json.dumps(preflight(config_path), ensure_ascii=False, indent=2, sort_keys=True))
        return
    proposer_override = (
        ResponseLedgerProposer(args.response_ledger, args.ledger_model)
        if args.response_ledger
        else None
    )
    result = run(config_path, args.procedure, proposer_override=proposer_override)
    print(
        json.dumps(
            {
                name: {
                    "heldout_recovery_count": value["heldout_recovery_count"],
                    "unique_composition_count": value["unique_composition_count"],
                    "compile_rate": value["compile_rate"],
                }
                for name, value in result.items()
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
