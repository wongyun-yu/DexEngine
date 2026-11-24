"""State of one text track (text.pipeline): the track.json keys (explicit, no defaults), the track root,
track_state.json load and save, the UTC clock, the network check (key or --dry) and the halt (STOP file)."""

from __future__ import annotations

import datetime as dt
import json
import types
from pathlib import Path

from dexengine.record.text import api as CTA
from dexengine.record.text.api import MARK, die

KEYS = (
    "pairs",
    "data_root",
    "frames",
    "views",
    "frame_size",
    "model",
    "redo_model",
    "text_only",
    "budget_usd",
    "submit_until",
    "min_batch",
    "retext",
    "redo_with_why",
    "clip_usd_cap",
    "round_cap",
    "via",
    "workers",
)


def track_root(track: str) -> Path:
    """Track root = run_collect_text/<track> under the data root (paths.main_root)."""
    from dexengine.core import paths as P  # noqa: PLC0415

    return P.run_path("run_collect_text", track=track, root=P.main_root())


def ns(**kw):
    return types.SimpleNamespace(**kw)


def load(root: Path) -> tuple[dict, dict]:
    """(track.json, track_state.json) of a track root; a missing key, an unknown key or an unknown model dies."""
    cfg = json.loads((root / "track.json").read_text())
    miss = [k for k in KEYS if k not in cfg]
    if miss:
        die(f"track.json is missing {miss} (no defaults)")
    extra = sorted(set(cfg) - set(KEYS))
    if extra:
        die(f"track.json has unknown keys {extra} (known: {', '.join(KEYS)})")
    if cfg["model"] not in CTA.PRICES or cfg["redo_model"] not in CTA.PRICES:
        die(f"unknown model {cfg['model']} · {cfg['redo_model']}")
    sf = root / "track_state.json"
    st = json.loads(sf.read_text()) if sf.is_file() else {"rounds": [], "retexted": {}}
    return cfg, st


def save(root: Path, st: dict) -> None:
    tmp = root / "track_state.json.tmp"
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    tmp.replace(root / "track_state.json")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def net_ok(dry: bool) -> str | None:
    """Why the network may not be used (None = it may)."""
    if dry:
        return "--dry"
    if not CTA.load_key():
        return f"no key ({CTA.KEY_FILE} and env ANTHROPIC_API_KEY both missing)"
    return None


def halt(root: Path, why: str) -> None:
    """Halt: write the reason to <root>/STOP and print an ALERT line."""
    (root / "STOP").write_text(f"{now_utc().isoformat()} {why}\n")
    print(f"{MARK} track ALERT halt — {why}", flush=True)
