#!/usr/bin/env python3
"""Derive a conservative evaluation checkpoint by shrinking PPO residual logits.

The active-population score is a fixed prior outside ``actor_cell``. Scaling the
final actor layer therefore multiplies only the learned correction and leaves
the transparent benchmark prior unchanged. Derived checkpoints are deliberately
marked evaluation-only because their saved AdamW moments no longer correspond
exactly to the transformed weights.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Mapping

import torch


ACTOR_KEYS = ("actor_cell.weight", "actor_cell.bias")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_payload(payload: Mapping, scale: float, parent_sha256: str) -> dict:
    result_scale = float(scale)
    if not 0.0 <= result_scale <= 1.0:
        raise ValueError("residual scale must be in [0, 1]")
    if not isinstance(payload, Mapping) or "policy_state_dict" not in payload:
        raise ValueError("source checkpoint is not a regional policy payload")
    if payload.get("checkpoint_derivation") is not None:
        raise ValueError("derive from the original trained checkpoint, not a derived one")

    result = copy.deepcopy(dict(payload))
    state = result["policy_state_dict"]
    missing = [key for key in ACTOR_KEYS if key not in state]
    if missing:
        raise ValueError(f"checkpoint is missing final actor tensors: {missing}")
    for key in ACTOR_KEYS:
        tensor = state[key]
        if not torch.is_tensor(tensor):
            raise ValueError(f"checkpoint tensor {key!r} is invalid")
        state[key] = tensor * result_scale

    result["training_resume_allowed"] = False
    result["checkpoint_derivation"] = {
        "method": "learned_residual_logit_shrinkage",
        "residual_scale": result_scale,
        "parent_checkpoint_sha256": str(parent_sha256),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "evaluation_only",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if source == output:
        raise ValueError("output must differ from the trained source checkpoint")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    derived = derive_payload(payload, args.scale, _sha256(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(derived, temporary)
    temporary.replace(output)
    print(
        f"[DERIVED CHECKPOINT] scale={float(args.scale):.6g} "
        f"sha256={_sha256(output)} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
