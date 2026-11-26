"""The entry decides the kit (dexengine.rl.app): train · judge never render, a render entry must render, the removed
switch dies, the removed renderer flags die with the pointer to the record render (dexengine.rl.render_removed), and the
context marker names the env count."""

from __future__ import annotations

import pytest

from dexengine.rl import app as TK
from dexengine.rl import render_removed as RR

KW = {"task": "train", "record_session": False, "pose_render": False, "enable_cameras": False}


@pytest.mark.parametrize("entry", TK.KIT_PLAIN)
def test_plain_entries_take_the_plain_kit_and_refuse_render_requests(entry):
    env = {TK.ENTRY_ENV: entry}
    assert TK.plain_kit(env, **KW) is True
    for bad in ({"record_session": True}, {"pose_render": True}, {"enable_cameras": True}):
        with pytest.raises(ValueError, match="plain kit"):
            TK.plain_kit(env, **{**KW, **bad})
    with pytest.raises(ValueError, match="plain kit"):
        TK.plain_kit({**env, TK.FROZEN_RENDER: "1"}, **KW)


def test_render_entries_must_render():
    assert TK.plain_kit({TK.ENTRY_ENV: "render", TK.FROZEN_RENDER: "1"}, **KW) is False
    assert TK.plain_kit({TK.ENTRY_ENV: "replay"}, **{**KW, "task": "eval"}) is False
    with pytest.raises(ValueError, match="nothing renders"):
        TK.plain_kit({TK.ENTRY_ENV: "render"}, **KW)


@pytest.mark.parametrize("flag", sorted(RR.ARGS))
def test_removed_renderer_flags_die_with_the_pointer(flag, capsys):
    """--render_per_clip_videos, --video, the --videos_* family …: the trainer parser dies naming the record render."""
    from dexengine.rl.args import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([flag, "1"])
    err = capsys.readouterr().err
    assert f"{RR.MARK} {flag} removed" in err and "scripts/stage4_record.py" in err


def test_removed_renderer_env_dies_with_the_pointer():
    assert RR.env_error({"DEXENGINE_KEY_LIGHT": "3200", "DEXENGINE_AUG_ROOM": ""}) == ""
    for k in ("DEXENGINE_TRAJ_RENDER", "DEXENGINE_AUG_FLOOR_MDL", "DEXENGINE_AUG_ANYTHING", "RENDER_EGO_FOCAL_MM"):
        err = RR.env_error({k: "x"})
        assert k in err and "scripts/stage4_record.py" in err


def test_entry_is_required_and_the_removed_switch_dies():
    with pytest.raises(ValueError, match="required"):
        TK.entry_of({})
    with pytest.raises(ValueError, match="removed"):
        TK.entry_of({TK.ENTRY_ENV: "train", "DEXENGINE_TRAIN_NO_RENDER": "1"})


def test_context_line_names_entry_kit_and_env_count():
    assert TK.context_line("render", False, 1) == "[RENDER_CTX] on · entry=render · kit=rendering · NE=1"
    assert TK.context_line("train", True, 4096).startswith("[RENDER_CTX] off · entry=train · kit=plain")
