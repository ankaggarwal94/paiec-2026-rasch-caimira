"""Verify every prepared release file and the original archive member identities."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        rel = PurePosixPath(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            failures.append({"path": entry["path"], "reason": "unsafe manifest path"})
            continue
        path = ROOT.joinpath(*rel.parts)
        if path.is_file() is False or path.is_symlink():
            failures.append({"path": entry["path"], "reason": "missing regular file"})
            continue
        data = path.read_bytes()
        if len(data) != entry["bytes"] or digest(data) != entry["sha256"]:
            failures.append({"path": entry["path"], "reason": "file identity differs"})

    archive = ROOT / "runtime/original_submission/18-s44-rasch-caimira.zip"
    expected = "23546eb90c7469fe7ea2704879775885f1b32e3277b4b7b3a1015aa1c7c6297b"
    if digest(archive.read_bytes()) != expected:
        failures.append({"path": str(archive.relative_to(ROOT)), "reason": "selected archive differs"})
    with zipfile.ZipFile(archive) as candidate:
        for member in candidate.infolist():
            if member.is_dir():
                continue
            rel = PurePosixPath(member.filename)
            if rel.is_absolute() or ".." in rel.parts:
                failures.append({"path": member.filename, "reason": "unsafe original member path"})
                continue
            extracted = ROOT / "runtime/candidate" / Path(*rel.parts)
            if extracted.read_bytes() != candidate.read(member):
                failures.append({"path": member.filename, "reason": "extracted member differs"})
    print(json.dumps({"status": "failed" if failures else "passed", "files_checked": len(manifest["files"]),
                      "selected_archive_sha256": expected, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
