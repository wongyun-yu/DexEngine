"""Session-log classification (session.launch.classify).

The fixture log is built from the emitters' marker constants, so a change to a marker that its checker reads
breaks these tests. Every marker the render checkers require gets one negative case (drop it -> "marks").
"""

from __future__ import annotations

import pytest

from dexengine.record.looks import look_d as LOOK
from dexengine.record.looks import look_guard as LG
from dexengine.record.looks import look_settle as LS
from dexengine.record.looks import rev as LREV
from dexengine.record.render import body as BODY
from dexengine.record.render import clip_session as RCL
from dexengine.record.render import idle as IDLE
from dexengine.record.render import visibility as VIS
from dexengine.record.rollout import recorder as REC
from dexengine.record.scene import extras as EXV
from dexengine.record.scene import extras_rules as OCC
from dexengine.record.scene import supports as SUP
from dexengine.record.session import launch as L
from dexengine.record.session import persist_kit as PK
from dexengine.record.session import runner as SR

DONE_R, DONE_T, DONE_C = "[COLLECT_RENDER] DONE", "[TRAJ_REC] DONE", f"{RCL.MARK} DONE"
TREE = f"{SR.MARK} tree=/x commit=0 branch=b"
HOOK_R = L.HOOK_LINE["render"] + " → render"
HOOK_T = L.HOOK_LINE["record"] + " → record"
CRASH = "[Fatal] [carb.crashreporter-breakpad.plugin] [crash] Crash detected in pid 1 thread 1"
CUDA = "[Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA GeForce RTX 5090"
MARKS = {  # one line per render checker (marker head -> line)
    "look": f"{LOOK.MARK} {LOOK.LOOK} applied · exposure ISO 1600",
    "look_token": f"{LG.MARK} canonical look {LOOK.LOOK} applied · stage anon:0",
    "settle": f"{LS.SETTLE_MARK} [0.3, 0.1] (tol 0.12)",
    "support": SUP.SHOWN + "0/0",
    "body": f"{BODY.MARK} none · root z +0.0000 m",
    "idle": f"{IDLE.MARK} no idle hand",
    "vis": f"{VIS.VIS_MARK} left eye 1 frame",
    "extras": EXV.marker("c", []),
    "overhang": OCC.overhang_marker("c", []),
    "head_view": f"{VIS.MARK} on · ego_left 1 frames",
    "cam_off": f"{RCL.OFF_MARK} · render product ['/Render/RP1']",
}
REC_ON = f"{REC.MARK} partner=on · 7 partner classes"


def lines(*xs: str) -> str:
    return "\n".join(xs) + "\n"


def full_render(done: str = DONE_R, drop: str = "") -> str:
    return lines(TREE, HOOK_R, *(v for k, v in MARKS.items() if k != drop), done)


def c(stage, rc, text, done):
    return L.classify(stage, rc, text, done)


def test_boot_before_hook():
    k, why = c("render", 0, lines(TREE, CUDA, CRASH), DONE_R)
    assert k == "boot" and L.BOOT_SIGNS[0][1] in why and L.BOOT_SIGNS[1][1] in why
    assert c("render", 139, lines(TREE), DONE_R)[0] == "boot"
    assert c("record", 124, lines(TREE), DONE_T)[0] == "boot"


def test_pre_hook_error_is_deterministic():
    k, why = c("record", 2, lines(TREE, "error: tree mismatch"), DONE_T)
    assert k == "pre" and "tree mismatch" in why


def test_native_crash_after_hook():
    k, why = c("render", 139, lines(TREE, HOOK_R, CRASH), DONE_R)
    assert k == "crash" and "rc 139" in why
    mutex = (
        "python3.11: Mutex.h:1: void carb::thread::detail::BaseMutex<false>::unlock(): Assertion "
        "(m_owner == pthread_self()) failed: unlock() called by non-owning thread"
    )
    assert c("render_clip", 134, lines(TREE, HOOK_R, mutex), DONE_C)[0] == "crash"
    assert c("render_clip", 134, full_render(DONE_C), DONE_C)[0] == "crash"  # abort after the end marker


def test_fail_after_hook():
    assert c("render", 1, lines(TREE, HOOK_R, CRASH), DONE_R)[0] == "fail"
    assert c("render", 137, lines(TREE, HOOK_R), DONE_R)[0] == "fail"
    assert c("render", 0, lines(TREE, HOOK_R), DONE_R)[0] == "fail"
    assert c("render", 1, full_render(), DONE_R)[0] == "fail"


def test_stall_line_wins():
    text = lines(TREE, HOOK_R, f"{L.STALL_LINE} — no progress for 600 s")
    k, why = c("render", -15, text, DONE_R)
    assert k == "stall" and "no progress" in why


def test_persist_worker_stall_line_is_a_stall():
    """session.persist writes its own stall line (Kit-reuse worker); it must start with launch.STALL_LINE."""
    import ast
    from pathlib import Path

    src = Path(L.__file__).with_name("persist.py").read_text()
    f = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.JoinedStr)
        and any(isinstance(v, ast.Constant) and "worker log did not grow" in v.value for v in n.values)
    ]
    assert len(f) == 1 and ast.unparse(f[0].values[0].value) == "_CLA.STALL_LINE"
    line = L.STALL_LINE + "".join(v.value if isinstance(v, ast.Constant) else "600" for v in f[0].values[1:])
    k, why = c("render_clip", 0, lines(TREE, HOOK_R, line), DONE_C)
    assert k == "stall" and "did not grow" in why


@pytest.mark.parametrize("stage,done", [("render", DONE_R), ("render_clip", DONE_C)])
def test_full_render_log_is_ok(stage, done):
    assert c(stage, 0, full_render(done), done) == ("ok", "")


def test_look_guard_token_marker_required():
    """A render log without the frame-guard token line (looks.look_guard.mark, set only by look_d.apply) is a
    canonical defect."""
    k, why = c("render", 0, full_render(drop="look_token"), DONE_R)
    assert k == "marks" and LG.MARK in why


@pytest.mark.parametrize("drop", sorted(MARKS))
def test_each_render_marker_required(drop):
    k, why = c("render", 0, full_render(drop=drop), DONE_R)
    assert k == "marks", drop


def test_brightness_gates_match_drawn_episodes():
    import ast
    from pathlib import Path

    from dexengine.record.render import writer as CW

    # the render's frame loop prints the head itself (render.episode_frames.capture_frames)
    ep = ast.parse((Path(CW.__file__).with_name("episode_frames.py")).read_text())
    assert any(
        isinstance(n, ast.JoinedStr)
        and isinstance(n.values[0], ast.FormattedValue)
        and ast.unparse(n.values[0].value) == "CW.FRAME_HEAD"
        for n in ast.walk(ep)
    )
    frame0 = f"{CW.FRAME_HEAD}0/10 (step 0)"
    k, why = c("render", 0, full_render().replace(DONE_R, frame0 + "\n" + DONE_R), DONE_R)
    assert k == "marks" and LREV.FLOOR_MARK in why
    gate = f"{LREV.FLOOR_MARK} D_exposure ok"
    assert c("render", 0, full_render().replace(DONE_R, frame0 + "\n" + gate + "\n" + DONE_R), DONE_R) == ("ok", "")


def test_render_clip_without_episodes_skips_per_episode_marks():
    resumed = PK.resume_line(4, 4)
    text = full_render(DONE_C).replace(MARKS["vis"] + "\n", "").replace(MARKS["head_view"] + "\n", "")
    assert c("render_clip", 0, text, DONE_C)[0] == "marks"
    assert c("render_clip", 0, text.replace(DONE_C, resumed + "\n" + DONE_C), DONE_C) == ("ok", "")
    assert c("render_clip", 0, text.replace(DONE_C, PK.resume_line(1, 4) + "\n" + DONE_C), DONE_C)[0] == "marks"


def test_record_stage_partner_marker():
    assert c("record", 0, lines(TREE, HOOK_T, REC_ON, DONE_T), DONE_T) == ("ok", "")
    k, why = c("record", 0, lines(TREE, HOOK_T, DONE_T), DONE_T)
    assert k == "marks" and "partner=on" in why


def test_unknown_stage_dies():
    with pytest.raises(SystemExit):
        c("bogus", 0, "", DONE_R)
