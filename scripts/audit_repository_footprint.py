from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_TOP = 15
DEFAULT_MIN_SIZE_MIB = 1.0


class GitCommandError(RuntimeError):
    """Raised when a read-only git query fails."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_git_bytes(
    repo_root: Path,
    args: Sequence[str],
    *,
    input_text: str | None = None,
) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repo_root), *args),
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=input_text is not None,
    )
    if completed.returncode != 0:
        stderr = (
            completed.stderr.decode("utf-8", errors="replace")
            if isinstance(completed.stderr, bytes)
            else completed.stderr
        )
        raise GitCommandError(f"git {' '.join(args)} failed: {stderr.strip()}")
    stdout = completed.stdout
    return stdout.encode("utf-8") if isinstance(stdout, str) else stdout


def _run_git_text(
    repo_root: Path,
    args: Sequence[str],
    *,
    input_text: str | None = None,
) -> str:
    return _run_git_bytes(repo_root, args, input_text=input_text).decode("utf-8", errors="surrogateescape")


def _split_nul(output: bytes) -> list[str]:
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in output.split(b"\0")
        if item
    ]


def _format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} B"
            if value < 10.0:
                return f"{value:.2f} {unit}"
            if value < 100.0:
                return f"{value:.1f} {unit}"
            return f"{value:.0f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def _bytes_from_mib(value: float) -> int:
    return int(value * 1024 * 1024)


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _file_record(path: str, size_bytes: int, *, object_id: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path,
        "size_bytes": size_bytes,
        "size_human": _format_size(size_bytes),
    }
    if object_id is not None:
        record["object_id"] = object_id
    return record


def _sort_and_limit(
    records: Iterable[dict[str, Any]],
    *,
    top: int,
    min_size_bytes: int,
) -> list[dict[str, Any]]:
    filtered = [record for record in records if int(record["size_bytes"]) >= min_size_bytes]
    filtered.sort(key=lambda record: (-int(record["size_bytes"]), str(record["path"])))
    return filtered[:top]


def _tracked_file_records(repo_root: Path, *, top: int, min_size_bytes: int) -> list[dict[str, Any]]:
    paths = _split_nul(_run_git_bytes(repo_root, ("ls-files", "-z")))
    records: list[dict[str, Any]] = []
    for path in paths:
        filesystem_path = repo_root / path
        try:
            stat_result = filesystem_path.lstat()
        except FileNotFoundError:
            continue
        records.append(_file_record(path, int(stat_result.st_size)))
    return _sort_and_limit(records, top=top, min_size_bytes=min_size_bytes)


def _untracked_file_records(repo_root: Path, *, top: int, min_size_bytes: int) -> list[dict[str, Any]]:
    paths = _split_nul(_run_git_bytes(repo_root, ("ls-files", "--others", "--exclude-standard", "-z")))
    records: list[dict[str, Any]] = []
    for path in paths:
        filesystem_path = repo_root / path
        try:
            stat_result = filesystem_path.lstat()
        except FileNotFoundError:
            continue
        if filesystem_path.is_dir():
            continue
        records.append(_file_record(path, int(stat_result.st_size)))
    return _sort_and_limit(records, top=top, min_size_bytes=min_size_bytes)


def _head_blob_records(repo_root: Path, *, top: int, min_size_bytes: int) -> list[dict[str, Any]]:
    output = _run_git_bytes(repo_root, ("ls-tree", "-rz", "-l", "--full-tree", "HEAD"))
    records: list[dict[str, Any]] = []
    for entry in _split_nul(output):
        try:
            metadata, path = entry.split("\t", 1)
        except ValueError:
            continue
        parts = metadata.split()
        if len(parts) != 4:
            continue
        _mode, object_type, object_id, size_text = parts
        if object_type != "blob" or size_text == "-":
            continue
        records.append(_file_record(path, int(size_text), object_id=object_id))
    return _sort_and_limit(records, top=top, min_size_bytes=min_size_bytes)


def _reachable_blob_records(repo_root: Path, *, top: int, min_size_bytes: int) -> list[dict[str, Any]]:
    rev_list_entries = _run_git_text(repo_root, ("rev-list", "--objects", "--all")).splitlines()
    object_paths: dict[str, str] = {}
    object_ids: list[str] = []
    for entry in rev_list_entries:
        object_id, _separator, path = entry.partition(" ")
        if object_id in object_paths:
            continue
        object_paths[object_id] = path
        object_ids.append(object_id)

    if not object_ids:
        return []

    batch_input = "".join(f"{object_id}\n" for object_id in object_ids)
    batch_output = _run_git_text(
        repo_root,
        ("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"),
        input_text=batch_input,
    )
    records: list[dict[str, Any]] = []
    for line in batch_output.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        object_id, object_type, size_text = parts
        if object_type != "blob":
            continue
        path = object_paths.get(object_id) or "<path unavailable>"
        records.append(_file_record(path, int(size_text), object_id=object_id))
    return _sort_and_limit(records, top=top, min_size_bytes=min_size_bytes)


def _parse_count_objects(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            parsed[key.strip()] = value.strip()
    return parsed


def _pack_summary(repo_root: Path) -> dict[str, Any]:
    pack_dir_text = _run_git_text(repo_root, ("rev-parse", "--git-path", "objects/pack")).strip()
    pack_dir = Path(pack_dir_text)
    if not pack_dir.is_absolute():
        pack_dir = repo_root / pack_dir
    pack_dir = pack_dir.resolve()

    pack_files: list[dict[str, Any]] = []
    for pack_file in sorted(pack_dir.glob("*.pack")):
        size_bytes = int(pack_file.stat().st_size)
        pack_files.append(
            {
                "path": _relative_or_absolute(pack_file, repo_root),
                "size_bytes": size_bytes,
                "size_human": _format_size(size_bytes),
            }
        )
    pack_files.sort(key=lambda record: (-int(record["size_bytes"]), str(record["path"])))
    pack_total_bytes = sum(int(record["size_bytes"]) for record in pack_files)

    return {
        "pack_dir": _relative_or_absolute(pack_dir, repo_root),
        "pack_count": len(pack_files),
        "pack_total_bytes": pack_total_bytes,
        "pack_total_human": _format_size(pack_total_bytes),
        "pack_files": pack_files,
        "count_objects": _parse_count_objects(_run_git_text(repo_root, ("count-objects", "-vH"))),
    }


def build_audit(repo_root: Path, *, top: int, min_size_bytes: int) -> dict[str, Any]:
    resolved_repo_root = repo_root.resolve()
    discovered_root = Path(_run_git_text(resolved_repo_root, ("rev-parse", "--show-toplevel")).strip()).resolve()
    if discovered_root != resolved_repo_root:
        resolved_repo_root = discovered_root

    return {
        "repo_root": str(resolved_repo_root),
        "parameters": {
            "top": top,
            "min_size_bytes": min_size_bytes,
            "min_size_human": _format_size(min_size_bytes),
        },
        "git_object_store": _pack_summary(resolved_repo_root),
        "tracked_large_files": _tracked_file_records(resolved_repo_root, top=top, min_size_bytes=min_size_bytes),
        "current_tree_large_blobs": _head_blob_records(resolved_repo_root, top=top, min_size_bytes=min_size_bytes),
        "reachable_history_large_blobs": _reachable_blob_records(
            resolved_repo_root,
            top=top,
            min_size_bytes=min_size_bytes,
        ),
        "untracked_large_files": _untracked_file_records(resolved_repo_root, top=top, min_size_bytes=min_size_bytes),
    }


def _format_record_lines(records: Sequence[dict[str, Any]]) -> list[str]:
    if not records:
        return ["  none"]
    lines: list[str] = []
    for record in records:
        object_id = record.get("object_id")
        suffix = f"  {str(object_id)[:12]}" if object_id else ""
        lines.append(f"  {record['size_human']:>10}  {record['path']}{suffix}")
    return lines


def format_text(audit: dict[str, Any]) -> str:
    params = audit["parameters"]
    object_store = audit["git_object_store"]
    lines = [
        "Repository footprint audit",
        f"repo_root: {audit['repo_root']}",
        f"threshold: >= {params['min_size_human']}; top: {params['top']}",
        "",
        ".git pack size",
        f"  pack_dir: {object_store['pack_dir']}",
        f"  pack_count: {object_store['pack_count']}",
        f"  pack_total: {object_store['pack_total_human']}",
    ]

    for pack_file in object_store["pack_files"]:
        lines.append(f"  {pack_file['size_human']:>10}  {pack_file['path']}")

    count_objects = object_store["count_objects"]
    if count_objects:
        lines.extend(["", "git count-objects -vH"])
        for key in ("count", "size", "in-pack", "packs", "size-pack", "prune-packable", "garbage", "size-garbage"):
            if key in count_objects:
                lines.append(f"  {key}: {count_objects[key]}")

    sections = (
        ("Tracked large files (working tree stat)", "tracked_large_files"),
        ("Current-tree large blobs (HEAD)", "current_tree_large_blobs"),
        ("Reachable-history large blobs (all refs)", "reachable_history_large_blobs"),
        ("Untracked large files (git-excluded files ignored)", "untracked_large_files"),
    )
    for title, key in sections:
        lines.extend(["", title])
        lines.extend(_format_record_lines(audit[key]))

    return "\n".join(lines) + "\n"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only repository footprint audit for release closeout. Reports tracked large files, "
            ".git pack size, current-tree blobs, reachable history blobs, and untracked large files."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root(),
        help="Repository root to audit. Defaults to the checkout containing this script.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. JSON is stable enough for release automation snapshots.",
    )
    parser.add_argument(
        "--top",
        type=_positive_int,
        default=DEFAULT_TOP,
        help=f"Maximum rows per large-file/blob section. Default: {DEFAULT_TOP}.",
    )
    parser.add_argument(
        "--min-size-mib",
        type=_nonnegative_float,
        default=DEFAULT_MIN_SIZE_MIB,
        help=f"Minimum file/blob size to report, in MiB. Default: {DEFAULT_MIN_SIZE_MIB}.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        audit = build_audit(args.repo_root, top=args.top, min_size_bytes=_bytes_from_mib(args.min_size_mib))
    except GitCommandError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        print(format_text(audit), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
