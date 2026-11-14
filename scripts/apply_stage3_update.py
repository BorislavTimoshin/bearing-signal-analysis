"""Install this additive patch after checking payload and project hashes.

Run from the unpacked update, NOT from a copied file in the target project.
Default is a dry run. Existing differing files are never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError(f"Unsafe path: {relative}")
    return path


def install(source, target, apply=False):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("Unpack update outside your project")
    manifest = json.loads(
        (source / "STAGE3_UPDATE_MANIFEST.json").read_text(encoding="utf-8")
    )
    errors, pending = [], []
    for name, expected in manifest["required_project_files"].items():
        path = inside(target, name)
        if not path.is_file() or sha(path) != expected:
            errors.append("BASE MISMATCH: " + name)
    for name, expected in manifest["payload"].items():
        src, dst = inside(source, name), inside(target, name)
        if not src.is_file() or sha(src) != expected:
            errors.append("UPDATE CORRUPT: " + name)
        elif dst.exists():
            if not dst.is_file() or sha(dst) != expected:
                errors.append("LOCAL CONFLICT (not overwritten): " + name)
        else:
            pending.append((src, dst))
    if errors:
        raise ValueError("\n".join(errors))
    if apply:
        created = []
        try:
            for src, dst in pending:
                dst.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation also protects against a new concurrent file.
                with dst.open("xb") as stream:
                    created.append(dst)
                    stream.write(src.read_bytes())
        except Exception:
            for path in reversed(created):
                path.unlink()
            raise
    return len(pending)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--apply", action="store_true", help="Apply after all checks pass"
    )
    args = parser.parse_args()
    try:
        count = install(Path(__file__).resolve().parents[1], args.project, args.apply)
    except ValueError as error:
        parser.exit(
            1, str(error) + "\nNo files copied. Resolve conflicts explicitly.\n"
        )
    print(
        ("ADDED " if args.apply else "DRY RUN: would add ")
        + str(count)
        + " files. No existing files overwritten."
    )
