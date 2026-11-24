"""Hard-exit guards of the Kit-reuse worker (session.persist_kit.render_clips): separate watcher processes that KILL
the worker when the app close hangs, the failure path hangs, or the owner dies. Standard library only."""

from __future__ import annotations

import os

from dexengine.record.session.persist import PERSIST_MARK

#: app close cap: the outputs and the end marker are done before the close. Measured: after 10 jobs app.close hung for
#: over 4 minutes in the replicator's stage close (AnnotatorCache.clear; cause not found, worked around). Past the cap
#: a watcher process leaves a marker and kill -9s (a separate process, not a Python thread, so it acts even when the
#: main thread holds the GIL).
CLOSE_GRACE_S = 60


#: hard-exit grace (s) on the failure path: the process ends even if the traceback, the failure line and the flush do
#: not finish in time (stdout, stderr or the main thread held by Kit).
DIE_GRACE_S = 30
#: owner watch period (s): when the owner dies, the worker ends within this time even mid-job or hung after a failure
#: (outside the idle polling of persist_kit._q_jobs).
OWNER_POLL_S = 5


def _reaper(sh: str) -> None:
    """The one home of the hard-exit watchers: a separate process (not a Python thread: it acts while the main thread
    is stuck in the GIL or Kit) in a new session (not swept by the caller's session reaping; when the worker ends first,
    kill -0 is false and the watcher ends itself)."""
    import subprocess

    subprocess.Popen(["sh", "-c", sh], start_new_session=True, stdin=subprocess.DEVNULL)


def arm_kill(after_s: float, why: str) -> None:
    """KILL this process if it is still alive after after_s (one marker line)."""
    me = os.getpid()
    _reaper(
        f"sleep {after_s}; kill -0 {me} 2>/dev/null && "
        f'echo "{PERSIST_MARK} {why} over {after_s} s: hard exit (KILL)" '
        f"&& kill -9 {me}"
    )


def watch_owner(owner: int) -> None:
    """KILL this worker when the owner pid dies, also mid-job or hung after a failure (the idle polling of
    persist_kit._q_jobs sees the owner only between jobs; in that gap a worker whose owner and queue were gone held
    its GPU until SERVER_TIMEOUT_S; test T1). owner 0 = no owner: no watch."""
    if owner <= 0:
        return
    me = os.getpid()
    _reaper(
        f"while kill -0 {owner} 2>/dev/null && kill -0 {me} 2>/dev/null; do sleep {OWNER_POLL_S}; done; "
        f'kill -0 {me} 2>/dev/null && echo "{PERSIST_MARK} owner {owner} died: worker {me} hard exit (KILL)" '
        f"&& kill -9 {me}"
    )


def close_bounded(app) -> None:
    arm_kill(CLOSE_GRACE_S, "app close (outputs and end marker are already done)")
    app.close()
