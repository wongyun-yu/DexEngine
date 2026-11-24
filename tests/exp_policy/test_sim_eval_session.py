"""E5 session plumbing: the run_scene option mapping and log checks, and the markers the batch/eval checks rely on
(producer text in the closed loop ↔ consumer patterns), checked without Isaac."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from experiments.sim_eval import batch as B
from experiments.sim_eval import run_scene as RS

SIM_EVAL = Path(__file__).resolve().parents[2] / "experiments" / "sim_eval"


def _fstring_text(path: Path, needle: str, mark: str) -> str:
    """The literal text of the one f-string in `path` that contains `needle`, with `{MARK}` filled in."""
    hits = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.JoinedStr):
            parts = [
                v.value if isinstance(v, ast.Constant) else ("{MARK}" if ast.unparse(v.value) == "MARK" else "<x>")
                for v in node.values
            ]
            text = "".join(parts)
            if needle in text:
                hits.append(text.replace("{MARK}", mark))
    assert len(hits) == 1, hits
    return hits[0]


def test_run_scene_options_and_log_rc(monkeypatch):
    opts, env, sw = RS.eval_options(
        {
            "profile": "official",
            "policy_until_t": True,
            "policy_job_text": True,
            "place_marker": None,
            "start_override": "/s.json",
            "render_overlay_root": "/o",
            "epstate": False,
            "all_obj_collision": True,
        }
    )
    assert opts == {
        "spawn_f0": "1",
        "extra_f0": "1",
        "profile": "official",
        "policy_until_t": "1",
        "policy_job_text": "1",
        "start_override": "/s.json",
    }
    assert env == {"DEXENGINE_RENDER_OVERLAY_ROOT": "/o"}
    assert sw == {"arm_ff_vel_scale": 0.0, "all_obj_collision": "1"}
    with pytest.raises(SystemExit):
        RS.eval_options({"profile": "legacy"})
    phys = "[E5_EVAL] [E5_ARMCTRL] policy\n[E5_EVAL] [E5_PHYS] x\n[E5_EVAL] [E5_MARKERS] y\n"
    done = "[E5_EVAL] DONE modes=2/2\n"
    assert RS.marker_rc(phys, ["gate", "policy"]) == 3
    assert RS.marker_rc(phys + done, ["gate", "policy"]) == 0
    assert RS.marker_rc(phys + done, ["policy", "replay"]) == 5


def test_markers_match_their_consumers():
    loop = SIM_EVAL / "closed_loop.py"
    na = _fstring_text(loop, "desk_stretch", "[E5_EVAL]")
    assert B.NA_WORLD_RE.search(f"Traceback …\nSystemExit: {na}\n")
    start = _fstring_text(SIM_EVAL / "loop" / "reset.py", "[E5_START] source=", "[E5_EVAL]")
    assert re.search(
        r"^\[E5_EVAL\] \[E5_START\] source=spider_",
        "[E5_EVAL] " + start.replace("source=<x>", "source=spider_start"),
        re.M,
    )
    assert "DONE modes=" in _fstring_text(loop, "DONE modes=", "[E5_EVAL]")
    src = loop.read_text() + "".join((SIM_EVAL / "loop" / f).read_text() for f in ("control.py", "episode.py"))
    for marker in ("[E5_ARMCTRL] ", "[E5_PHYS] ", "[E5_MARKERS] "):
        assert f'"{marker}' in src or f'f"{marker}' in src, marker
