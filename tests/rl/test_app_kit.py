"""The entry decides the kit (dexengine.rl.app): train · judge never render, a render entry must render, the removed
switch and the producerless viewport recorder die, and the context marker names the env count."""

from __future__ import annotations

import pytest

from dexengine.rl import app as TK

KW = {"task": "train", "video": False, "per_clip_videos": False, "enable_cameras": False}


@pytest.mark.parametrize("entry", TK.KIT_PLAIN)
def test_plain_entries_take_the_plain_kit_and_refuse_render_requests(entry):
    env = {TK.ENTRY_ENV: entry}
    assert TK.plain_kit(env, **KW) is True
    for bad in ({"per_clip_videos": True}, {"enable_cameras": True}):
        with pytest.raises(ValueError, match="plain kit"):
            TK.plain_kit(env, **{**KW, **bad})
    with pytest.raises(ValueError, match="plain kit"):
        TK.plain_kit({**env, TK.FROZEN_RENDER: "1"}, **KW)
    with pytest.raises(ValueError, match="plain kit"):
        TK.plain_kit({**env, TK.TRAJ_RENDER: "out"}, **KW)


def test_render_entries_must_render():
    assert TK.plain_kit({TK.ENTRY_ENV: "render", TK.FROZEN_RENDER: "1"}, **KW) is False
    assert TK.plain_kit({TK.ENTRY_ENV: "replay"}, **{**KW, "task": "eval"}) is False
    with pytest.raises(ValueError, match="nothing renders"):
        TK.plain_kit({TK.ENTRY_ENV: "render"}, **KW)


@pytest.mark.parametrize("entry", TK.ENTRIES)
def test_the_viewport_recorder_dies_everywhere(entry):
    with pytest.raises(ValueError, match="no producer"):
        TK.plain_kit({TK.ENTRY_ENV: entry, TK.FROZEN_RENDER: "1"}, **{**KW, "video": True})


def test_entry_is_required_and_the_removed_switch_dies():
    with pytest.raises(ValueError, match="required"):
        TK.entry_of({})
    with pytest.raises(ValueError, match="removed"):
        TK.entry_of({TK.ENTRY_ENV: "train", "DEXENGINE_TRAIN_NO_RENDER": "1"})


def test_context_line_names_entry_kit_and_env_count():
    assert TK.context_line("render", False, 1) == "[RENDER_CTX] on · entry=render · kit=rendering · NE=1"
    assert TK.context_line("train", True, 4096).startswith("[RENDER_CTX] off · entry=train · kit=plain")
