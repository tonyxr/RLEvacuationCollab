#!/usr/bin/env python3
"""Content-addressed cache for complete, reproducible PPO policy artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence


SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(root: str | Path, paths: Sequence[str]) -> dict[str, str]:
    root = Path(root).resolve()
    return {
        str(relative): sha256_file(root / relative)
        for relative in sorted(str(value) for value in paths)
    }


def canonical_contract_key(contract: Mapping) -> str:
    encoded = json.dumps(
        dict(contract),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PolicyCache:
    """Store or restore a checkpoint only under an exact semantic contract."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def _entry(self, contract: Mapping) -> tuple[str, Path, Path, Path]:
        key = canonical_contract_key(contract)
        directory = self.root / key[:2] / key
        return key, directory, directory / "regional_policy.pt", directory / "manifest.json"

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def restore(
        self,
        contract: Mapping,
        destination: str | Path,
        *,
        required_metadata: Mapping | None = None,
    ) -> dict | None:
        key, _, checkpoint, manifest_path = self._entry(contract)
        if not checkpoint.exists() or not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            int(manifest.get("schema_version", -1)) != SCHEMA_VERSION
            or manifest.get("cache_key") != key
            or manifest.get("contract") != dict(contract)
        ):
            raise RuntimeError(f"Policy-cache manifest contract is invalid: {manifest_path}")
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Policy-cache metadata is invalid: {manifest_path}")
        if required_metadata is not None and any(
            metadata.get(name) != expected
            for name, expected in required_metadata.items()
        ):
            return None
        digest = sha256_file(checkpoint)
        if digest != manifest.get("checkpoint_sha256"):
            raise RuntimeError(f"Policy-cache checkpoint checksum failed: {checkpoint}")
        destination = Path(destination)
        self._atomic_copy(checkpoint, destination)
        return {
            "status": "restored",
            "cache_key": key,
            "cache_checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "destination": str(destination.resolve()),
            "metadata": metadata,
        }

    def store(
        self,
        contract: Mapping,
        source: str | Path,
        *,
        metadata: Mapping | None = None,
    ) -> dict:
        key, directory, checkpoint, manifest_path = self._entry(contract)
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(f"Cannot cache missing policy checkpoint: {source}")
        directory.mkdir(parents=True, exist_ok=True)
        self._atomic_copy(source, checkpoint)
        digest = sha256_file(checkpoint)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "cache_key": key,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "contract": dict(contract),
            "metadata": {} if metadata is None else dict(metadata),
            "checkpoint_sha256": digest,
            "checkpoint_path": str(checkpoint),
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        return {
            "status": "stored",
            "cache_key": key,
            "cache_checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "source": str(source.resolve()),
        }
