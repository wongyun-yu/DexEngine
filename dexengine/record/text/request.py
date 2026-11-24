"""One clip's Messages API request: the role-user message (dataset note, metadata JSON with field provenance, frames,
retry note) and the request parameters (cached system prompt, structured JSON output)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from dexengine.record.text.api import die
from dexengine.record.text.check import SCHEMA, api_schema

#: Provenance of the metadata fields (ds = dataset, measured, image = render); prompt rule 14 ranks ds -> measured ->
#: image. Model input.
PROVENANCE = {
    "source": "ds",
    "objects": "ds",
    "hands": "ds",
    "scene": "ds",
    "segment": "ds",
    "motion": "measured",
    "phases": "measured",
    "frames": "image",
}


def user_text(rec: dict, ds_guide: dict) -> str:
    """The text part of the role-user message (dataset note, clip metadata JSON, frame note, retry note)."""
    g = ds_guide.get(rec["dataset"])
    if g is None:
        die(f"datasets.json has no note for dataset {rec['dataset']!r}")
    show = {
        k: rec[k]
        for k in ("dataset", "clip", "segment", "source", "objects", "hands", "scene", "motion", "phases")
        if rec.get(k) is not None
    }
    show["provenance"] = {k: v for k, v in PROVENANCE.items() if k in show or (k == "frames" and rec.get("frames"))}
    fr = (
        f"\n\nThe {len(rec['frames'])} images above are time-ordered frames of this clip's rendering "
        "(views: exo = fixed outside camera, egoL = head camera). They show a robot replaying the human motion; "
        "use them to identify the objects and the action, and still never mention the robot."
        if rec.get("frames")
        else ""
    )
    return (
        "Dataset note:\n"
        + g["note"]
        + "\n\nClip metadata (JSON):\n"
        + json.dumps(show, ensure_ascii=False, separators=(", ", ": "))
        + fr
        + retry_note(rec)
        + "\n\nReturn the fields for this clip as one JSON object."
    )


def retry_note(rec: dict) -> str:
    """Retry reasons of a regenerated clip (checker wording, built by retry); empty for a first request."""
    why = rec.get("retry_why") or []
    return (
        (
            "\n\nEarlier answers for this clip failed these automatic checks (Korean checker notes; '→' a "
            "passing form). Write a new answer that passes all of them:\n- " + "\n- ".join(why)
        )
        if why
        else ""
    )


def user_content(rec: dict, ds_guide: dict):
    """A string without frames; with frames [frame label, image (base64 JPEG)]... + the text (images before the
    text, as the docs recommend)."""
    text = user_text(rec, ds_guide)
    if not rec.get("frames"):
        return text
    blocks = []
    for k, f in enumerate(rec["frames"]):
        data = base64.b64encode(Path(f["path"]).read_bytes()).decode()
        ph = f" · {f['phase']} {f['at']}" if f.get("phase") else ""
        blocks += [
            {"type": "text", "text": f"Frame {k + 1} ({f['view']}, t={f['t_s']} s{ph}):"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
        ]
    return blocks + [{"type": "text", "text": text}]


def request_params(rec: dict, model: str, system: str, ds_guide: dict, max_tokens: int = 2400) -> dict:
    """Messages API params of one clip (cached system prompt, structured JSON output)."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "output_config": {"format": {"type": "json_schema", "schema": api_schema(SCHEMA)}},
        "messages": [{"role": "user", "content": user_content(rec, ds_guide)}],
    }
