"""JSON Dixon–Coles artefacts: versioned save/load, no pickle."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR
from src.models.dixon_coles import DixonColesModel

_LATEST_POINTER = "latest"
_MODEL_JSON = "model.json"
_META_JSON = "meta.json"


# Public rows that sum the score table, with no separate 1X2 calibration.
PUBLISHED_GRID_SUFFIX = "-grid"


def build_model_version(as_of: pd.Timestamp, content_hash: str) -> str:
    """Return ``dc-YYYYMMDD-<8char>`` from the training cutoff and content hash."""
    ts = pd.Timestamp(as_of) if as_of is not None else pd.NaT
    date_part = "00000000" if pd.isna(ts) else ts.strftime("%Y%m%d")
    return f"dc-{date_part}-{content_hash[:8]}"


def published_model_version(artefact_version: str) -> str:
    """Name written on a ledger row whose markets are plain score-table sums.

    The suffix is how a reader tells these rows from the 17 Nov publish, which
    applied a home/draw/away calibration and did not store a score.
    """
    if artefact_version.endswith(PUBLISHED_GRID_SUFFIX):
        return artefact_version
    return f"{artefact_version}{PUBLISHED_GRID_SUFFIX}"


def teams_in_model(model: DixonColesModel) -> set[str]:
    """Teams actually in the fit (attack keys). Publish must not invent others."""
    return set(model.attack.keys())


def save_artefact(
    model: DixonColesModel,
    temperature: float,
    hyperparams: dict,
    models_dir: Path | None = None,
) -> tuple[str, Path]:
    """Write ``models_dir/<version>/model.json`` and a ``latest`` pointer."""
    root = MODELS_DIR if models_dir is None else Path(models_dir)
    payload = _model_payload(model, float(temperature), dict(hyperparams))
    content_hash = _content_hash(payload)
    version = build_model_version(model.as_of, content_hash)
    artefact_dir = root / version
    artefact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artefact_dir / _MODEL_JSON
    model_path.write_text(_pretty_json(payload), encoding="utf-8")

    meta = {
        "model_version": version,
        "content_hash": content_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (artefact_dir / _META_JSON).write_text(_pretty_json(meta), encoding="utf-8")
    (root / _LATEST_POINTER).write_text(version + "\n", encoding="utf-8")
    return version, artefact_dir


def load_artefact(
    model_version: str | None,
    models_dir: Path | None = None,
) -> tuple[DixonColesModel, float, dict]:
    """Load a named artefact, or the latest via pointer / lexicographic ``dc-*``."""
    root = MODELS_DIR if models_dir is None else Path(models_dir)
    version = _resolve_version(model_version, root)
    model_path = root / version / _MODEL_JSON
    if not model_path.is_file():
        raise FileNotFoundError(f"model artefact not found: {version} in {root}")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    as_of_raw = payload.get("as_of")
    as_of = None if as_of_raw in (None, "") else pd.Timestamp(as_of_raw)
    model = DixonColesModel(
        attack=_float_map(payload.get("attack") or {}),
        defence=_float_map(payload.get("defence") or {}),
        home_advantage=float(payload["home_advantage"]),
        rho=float(payload["rho"]),
        xi=float(payload["xi"]),
        as_of=as_of,
        prior_kappa=float(payload.get("prior_kappa", 0.0) or 0.0),
        new_club_attack=float(payload.get("new_club_attack", 1.0) or 1.0),
        new_club_defence=float(payload.get("new_club_defence", 1.0) or 1.0),
    )
    temperature = float(payload["temperature"])
    hyperparams = dict(payload.get("hyperparams") or {})
    return model, temperature, hyperparams


def _model_payload(
    model: DixonColesModel, temperature: float, hyperparams: dict
) -> dict:
    return {
        "attack": _float_map(model.attack),
        "defence": _float_map(model.defence),
        "home_advantage": float(model.home_advantage),
        "rho": float(model.rho),
        "xi": float(model.xi),
        "as_of": _as_of_to_json(model.as_of),
        "prior_kappa": float(getattr(model, "prior_kappa", 0.0)),
        "new_club_attack": float(getattr(model, "new_club_attack", 1.0)),
        "new_club_defence": float(getattr(model, "new_club_defence", 1.0)),
        "temperature": float(temperature),
        "hyperparams": hyperparams,
    }


def _as_of_to_json(as_of: pd.Timestamp | None) -> str | None:
    if as_of is None:
        return None
    ts = pd.Timestamp(as_of)
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _float_map(values: dict) -> dict[str, float]:
    return {str(team): float(strength) for team, strength in values.items()}


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _pretty_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _content_hash(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _resolve_version(model_version: str | None, models_dir: Path) -> str:
    if model_version:
        return model_version
    pointer = models_dir / _LATEST_POINTER
    if pointer.is_file():
        name = pointer.read_text(encoding="utf-8").strip()
        if name and (models_dir / name / _MODEL_JSON).is_file():
            return name
    if not models_dir.is_dir():
        raise FileNotFoundError(f"no model artefacts in {models_dir}")
    versions = sorted(
        path.name
        for path in models_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("dc-")
        and (path / _MODEL_JSON).is_file()
    )
    if not versions:
        raise FileNotFoundError(f"no model artefacts in {models_dir}")
    return versions[-1]
