"""Verify packaged bytes against the release checksum list."""

from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
expected = json.loads((root / "RELEASE_SHA256.json").read_text())
for name, digest in expected.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise ValueError(name)
print(f"PASS: {len(expected)} file hashes")
