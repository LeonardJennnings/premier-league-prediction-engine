"""Read and append the three ledger logs. Existing lines are never rewritten."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config import LEDGER_DIR

PREDICTIONS_NAME = "predictions.jsonl"
BOOK_LINES_NAME = "book_lines.jsonl"
SETTLEMENTS_NAME = "settlements.jsonl"
MANIFEST_NAME = "manifest.json"
_LOG_NAMES = (PREDICTIONS_NAME, BOOK_LINES_NAME, SETTLEMENTS_NAME)


class LedgerRefused(RuntimeError):
    """The batch was refused. No ledger line from this call was written."""


def ledger_dir(path: Path | None = None) -> Path:
    return LEDGER_DIR if path is None else Path(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not a JSON object") from exc
    return rows


def ensure_ledger(path: Path | None = None) -> Path:
    root = ledger_dir(path)
    root.mkdir(parents=True, exist_ok=True)
    for name in _LOG_NAMES:
        file_path = root / name
        if not file_path.exists():
            file_path.write_bytes(b"")
    write_manifest(root)
    return root


def append_predictions(
    lines: list[dict],
    *,
    now_utc: datetime,
    path: Path | None = None,
) -> dict:
    """Append new prediction lines. A started match refuses the whole call.

    Identity is ``(match_id, forecast_line)``. A newer model does not add a
    second line. Lines already stored are skipped and do not trip the kickoff
    check, so a later run can still publish other matches.
    """
    root = ledger_dir(path)
    existing = read_jsonl(root / PREDICTIONS_NAME)
    keys = {
        (int(row["match_id"]), str(row["forecast_line"])) for row in existing
    }
    fresh: list[dict] = []
    skipped = 0
    for line in lines:
        key = (int(line["match_id"]), str(line["forecast_line"]))
        if key in keys:
            skipped += 1
            continue
        fresh.append(line)
        keys.add(key)
    started = [
        line
        for line in fresh
        if _parse_utc(line["kickoff_utc"]) <= _as_utc(now_utc)
    ]
    if started:
        listed = ", ".join(
            f"{line['match_id']} at {line['kickoff_utc']}" for line in started
        )
        raise LedgerRefused(
            f"Refusing to write a prediction after kickoff: {listed}"
        )
    if fresh:
        _append_jsonl(root / PREDICTIONS_NAME, fresh)
        write_manifest(root)
    return {
        "inserted": len(fresh),
        "skipped_duplicate": skipped,
        "inserted_ids": [int(line["match_id"]) for line in fresh],
    }


def append_historical_predictions(lines: list[dict], *, path: Path | None = None) -> dict:
    """Append rows that were published before kickoff, even if kickoff has passed.

    Used for the 17 Nov copy. A row whose publish time is not before kickoff
    is refused and nothing from this call is written.
    """
    late = [
        line
        for line in lines
        if str(line["published_at"]) >= str(line["kickoff_utc"])
    ]
    if late:
        listed = ", ".join(str(line["match_id"]) for line in late)
        raise LedgerRefused(
            f"Refusing historical rows that were not published before kickoff: {listed}"
        )
    root = ledger_dir(path)
    existing = read_jsonl(root / PREDICTIONS_NAME)
    keys = {
        (int(row["match_id"]), str(row["forecast_line"])) for row in existing
    }
    fresh: list[dict] = []
    skipped = 0
    for line in lines:
        key = (int(line["match_id"]), str(line["forecast_line"]))
        if key in keys:
            skipped += 1
            continue
        keys.add(key)
        fresh.append(line)
    if fresh:
        _append_jsonl(root / PREDICTIONS_NAME, fresh)
        write_manifest(root)
    return {
        "inserted": len(fresh),
        "skipped_duplicate": skipped,
        "inserted_ids": [int(line["match_id"]) for line in fresh],
    }


def append_book_lines(lines: list[dict], *, path: Path | None = None) -> dict:
    """Append opening or closing book prices. The same capture is not repeated."""
    return _append_keyed(
        BOOK_LINES_NAME,
        lines,
        key_fn=lambda row: (
            int(row["match_id"]),
            str(row["forecast_line"]),
            str(row["kind"]),
        ),
        path=path,
    )


def append_settlements(lines: list[dict], *, path: Path | None = None) -> dict:
    """Append result lines. One result per match. A correction is a later line."""
    root = ledger_dir(path)
    existing = read_jsonl(root / SETTLEMENTS_NAME)
    result_keys = {
        (int(row["match_id"]), str(row["forecast_line"]))
        for row in existing
        if row.get("kind") == "result"
    }
    correction_keys = {
        (
            int(row["match_id"]),
            str(row["forecast_line"]),
            row.get("home_goals"),
            row.get("away_goals"),
        )
        for row in existing
        if row.get("kind") == "correction"
    }
    status_keys = {
        (
            int(row["match_id"]),
            str(row["forecast_line"]),
            str(row.get("status")),
            str(row.get("kickoff_utc") or ""),
        )
        for row in existing
        if row.get("kind") == "status"
    }
    fresh: list[dict] = []
    skipped = 0
    for line in lines:
        kind = str(line["kind"])
        match_key = (int(line["match_id"]), str(line["forecast_line"]))
        if kind == "result":
            if match_key in result_keys:
                skipped += 1
                continue
            result_keys.add(match_key)
        elif kind == "correction":
            corr_key = (
                match_key[0],
                match_key[1],
                line.get("home_goals"),
                line.get("away_goals"),
            )
            if corr_key in correction_keys:
                skipped += 1
                continue
            correction_keys.add(corr_key)
        elif kind == "status":
            # A later kickoff is a new line. The same status and kickoff is not.
            status_key = (
                match_key[0],
                match_key[1],
                str(line.get("status")),
                str(line.get("kickoff_utc") or ""),
            )
            if status_key in status_keys:
                skipped += 1
                continue
            status_keys.add(status_key)
        else:
            raise ValueError(f"Unknown settlement kind: {kind}")
        fresh.append(line)
    if fresh:
        _append_jsonl(root / SETTLEMENTS_NAME, fresh)
        write_manifest(root)
    return {"inserted": len(fresh), "skipped_duplicate": skipped}


def write_manifest(path: Path | None = None) -> dict:
    """Regenerate counts and file hashes. Prediction lines are not edited."""
    root = ledger_dir(path)
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for name in _LOG_NAMES:
        file_path = root / name
        raw = file_path.read_bytes() if file_path.is_file() else b""
        files[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "lines": _line_count(raw),
        }
    versions = sorted(
        {
            str(row["model_version"])
            for row in read_jsonl(root / PREDICTIONS_NAME)
            if row.get("model_version")
        }
    )
    target = root / MANIFEST_NAME
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if target.is_file():
        try:
            previous = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = None
        # A second run with the same lines must not dirty the commit clock.
        if (
            isinstance(previous, dict)
            and previous.get("files") == files
            and previous.get("model_versions") == versions
            and previous.get("generated_at")
        ):
            generated_at = str(previous["generated_at"])
    manifest = {
        "files": files,
        "model_versions": versions,
        "generated_at": generated_at,
    }
    _write_bytes(target, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def _append_keyed(name: str, lines: list[dict], *, key_fn, path: Path | None) -> dict:
    root = ledger_dir(path)
    existing = read_jsonl(root / name)
    keys = {key_fn(row) for row in existing}
    fresh: list[dict] = []
    skipped = 0
    for line in lines:
        key = key_fn(line)
        if key in keys:
            skipped += 1
            continue
        keys.add(key)
        fresh.append(line)
    if fresh:
        _append_jsonl(root / name, fresh)
        write_manifest(root)
    return {"inserted": len(fresh), "skipped_duplicate": skipped}


def _append_jsonl(path: Path, lines: list[dict]) -> None:
    """Add lines. The previous file bytes stay an exact prefix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_bytes() if path.is_file() else b""
    if existing and not existing.endswith(b"\n"):
        existing += b"\n"
    extra = "".join(_dumps(line) for line in lines).encode("utf-8")
    payload = existing + extra
    if not payload.startswith(existing):
        raise RuntimeError(f"Refusing to rewrite {path}")
    _write_bytes(path, payload)


def _dumps(line: dict) -> str:
    return json.dumps(line, sort_keys=True, ensure_ascii=True) + "\n"


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _line_count(raw: bytes) -> int:
    if not raw.strip():
        return 0
    return len([line for line in raw.decode("utf-8").splitlines() if line.strip()])


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_utc(now_utc: datetime) -> datetime:
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=timezone.utc)
    return now_utc.astimezone(timezone.utc)
