"""Verify release bytes separately from historical experiment provenance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "PUBLIC_SHA256.json"


def verify(path: Path) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    hashes = json.loads(MANIFEST.read_text())
    if relative not in hashes:
        raise ValueError(f"File is not listed in the public integrity manifest: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != hashes[relative]:
        raise ValueError(f"Published file differs from its release hash: {relative}")
