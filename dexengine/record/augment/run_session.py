"""Augment rollout session start (record.augment.run.run_spec calls run_session): one Isaac session run, relaunched
once after a boot crash (start_retry rule), with the evidence printed per try (first exception, crash frames, Kit
boots on the node, shared-FS caches, GPU name, node). Part of the augment cache fingerprint (AUG_CODE_FILES)."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from dexengine.record.session import launch as CLA

#: relaunch after a boot crash: a workaround (root cause unknown). Rule = boot crash alone (not an rc list): when the
#: first attempt has rc != 0, 0 place marks and a session < START_RETRY_S, it is relaunched once in the same pod. A
#: second identical result stays FAIL (verdict by session_ok). Cost: relaunch <= 2 min; a deterministic failure shows
#: the same form twice, so one attempt is lost. The `[AUG_START_RETRY]` line prints the values that split the cases
#: (rc, first exception line, node, GPU kind, Kit boots on the same node within 60 s, cache places and file systems,
#: top crash frame libraries).
#: The relaunch goes once the cause of the boot crash is fixed.
START_RETRY = "[AUG_START_RETRY]"


START_RETRY_S = 120.0


#: kit60 window (± seconds), separate from the relaunch threshold
KIT_WIN_S = 60.0


#: Kit processes on the same node = processes that loaded carb (libcarb.so in /proc/<pid>/maps), within the /proc
#: view (the pod's namespace for a pod) · processes whose maps cannot be read (other uids) are counted apart
_KIT_LIB = b"libcarb.so"


_FRAME = re.compile(r"^\s*\d{3}:\s+([^\s!]+)!", re.M)


#: first exception line: the last line of a Python exception (`XxxError: …` · `XxxException: …`) or a fatal signal line
_EXC = re.compile(r"^(?:[\w.]+(?:Error|Exception|Exit|Interrupt)\b.*|Fatal Python error: .*)$", re.M)


_SHARED_FS = ("nfs", "lustre", "beegfs", "gpfs", "cifs", "ceph", "fuse", "wekafs", "panfs")


def start_retry(rc: int, places: int, elapsed_s: float, attempt: int) -> bool:
    """Is it a boot crash (relaunch)? Only on the first attempt with rc != 0, 0 place marks and a session <
    START_RETRY_S."""
    return attempt == 1 and rc != 0 and places == 0 and elapsed_s < START_RETRY_S


def first_exc(text: str) -> str:
    """First exception line of the log (_EXC); 'none' without one · cut when long."""
    m = _EXC.search(text)
    return m.group(0).strip()[:200] if m else "none"


def crash_frames(text: str, n: int = 3) -> str:
    """The top n crash report frame libraries (`NNN: lib!sym` line order); 'none' without them."""
    libs = _FRAME.findall(text)
    return ">".join(libs[:n]) if libs else "none"


def _kit_starts(t0: float, win: float = KIT_WIN_S) -> str:
    """Kit processes booted within t0 ± win s / all visible Kit (/proc view); '?(reason)' when not measurable."""
    try:
        hz = os.sysconf("SC_CLK_TCK")
        boot = time.time() - float(Path("/proc/uptime").read_text().split()[0])
        near = seen = unread = 0
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                kit = _KIT_LIB in (d / "maps").read_bytes()
            except PermissionError:
                unread += 1
                continue
            except OSError:
                continue
            if not kit:
                continue
            try:
                st = (d / "stat").read_text()
                start = boot + int(st[st.rindex(")") + 2 :].split()[19]) / hz
            except (OSError, ValueError, IndexError):
                continue
            seen += 1
            near += abs(start - t0) <= win
        return f"{near}/{seen}" + (f"(maps unreadable {unread})" if unread else "")
    except (OSError, ValueError) as e:
        return f"?({type(e).__name__})"


def _fstype(path: str) -> str:
    """File system of the mount holding path (longest /proc/mounts prefix), with '(shared)' for shared ones."""
    try:
        best, fs = "", "?"
        rp = os.path.realpath(path)
        for ln in Path("/proc/mounts").read_text().splitlines():
            f = ln.split()
            if len(f) > 2 and (rp == f[1] or rp.startswith(f[1].rstrip("/") + "/")) and len(f[1]) >= len(best):
                best, fs = f[1], f[2]
        return fs + ("(shared)" if fs.startswith(_SHARED_FS) else "")
    except OSError as e:
        return f"?({type(e).__name__})"


def _caches(env: dict) -> str:
    """Kit, ov and CUDA cache places and their file systems: XDG_CACHE_HOME/ov · ~/.local/share/ov · CUDA_CACHE_PATH."""
    home = env.get("HOME") or os.path.expanduser("~")
    xdg = env.get("XDG_CACHE_HOME") or f"{home}/.cache"
    places = {
        "ov": f"{xdg}/ov",
        "ovdata": f"{home}/.local/share/ov/data",
        "cuda": env.get("CUDA_CACHE_PATH") or f"{home}/.nv/ComputeCache",
    }
    return " ".join(f"{k}={v}@{_fstype(v if os.path.exists(v) else os.path.dirname(v))}" for k, v in places.items())


def _gpu_name(env: dict) -> str:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                "-i",
                (env.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0],
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return r.stdout.strip().replace(" ", "_") or f"?(rc {r.returncode})"
    except (OSError, subprocess.SubprocessError) as e:
        return f"?({type(e).__name__})"


def _node(env: dict) -> str:
    return env.get("SLURMD_NODENAME") or env.get("NODE_NAME") or os.uname().nodename


def _run_once(cmd: list[str], cwd: Path, env: dict, log: Path, stall_s: float) -> int:
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
        stalled = CLA.watch_stall(p, log, "record", stall_s)
        if stalled:
            f.write("\n" + stalled + "\n")
            f.flush()
    return p.returncode


def run_session(
    cmd: list[str], cwd: Path, env: dict, log: Path, stall_s: float = CLA.STALL_S, info: dict | None = None
) -> int:
    """One session: when the progress line ([TRAJ_REC] ) watch (session.launch.watch_stall, the same watch as the
    stage-1 record) reaps a stalled session it writes the stall line at the end of the log. Close skipping
    (augment.session close_skip) is the root side; this is the safety net above it. On a boot crash (start_retry) the
    first log stays as <log>.start1 and it relaunches once (marker START_RETRY); the verdict uses the last attempt's
    log (log). With info, start_retry = {rc, session_s, frames, node, gpu, kit60, caches, exc, rc2, session2_s, again}
    is added."""
    t0 = time.time()
    rc = _run_once(cmd, cwd, env or os.environ, log, stall_s)
    el = time.time() - t0
    txt = log.read_text(errors="replace")
    if not start_retry(rc, txt.count("[AUG_PLACE]"), el, 1):
        return rc
    e = env or dict(os.environ)
    meta = {
        "rc": rc,
        "session_s": round(el, 1),
        "exc": first_exc(txt),
        "frames": crash_frames(txt),
        "node": _node(e),
        "gpu": _gpu_name(e),
        "kit60": _kit_starts(t0),
        "caches": _caches(e),
    }
    first = log.with_name(log.name + ".start1")
    os.replace(log, first)
    print(
        f"{START_RETRY} boot crash rc {rc} · session {el:.1f} s < {START_RETRY_S:g} · [AUG_PLACE] 0 -> relaunch "
        f"once in the same pod "
        f"(workaround, root cause unknown) · first exception {meta['exc']!r} · node={meta['node']} gpu={meta['gpu']} "
        f"kit60={meta['kit60']} (±60 s boots/visible Kit) · "
        f"frames={meta['frames']} · {meta['caches']} · first log {first}",
        flush=True,
    )
    rc2 = _run_once(cmd, cwd, env or os.environ, log, stall_s)
    el2 = time.time() - t0 - el
    txt2 = log.read_text(errors="replace")
    again = rc2 != 0 and txt2.count("[AUG_PLACE]") == 0 and el2 < START_RETRY_S
    meta.update(rc2=rc2, session2_s=round(el2, 1), again=again, exc2=first_exc(txt2))
    print(
        f"{START_RETRY} second rc {rc2} · session {meta['session2_s']} s"
        f"{': the same boot crash, left FAIL · first exception ' + repr(meta['exc2']) if again else ''}",
        flush=True,
    )
    if info is not None:
        info["start_retry"] = meta
    return rc2
