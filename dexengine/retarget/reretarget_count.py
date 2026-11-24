"""The count rule of stage-1 retargeting (`reretarget`): every input clip without output must be explained by the
exclusion ledger; expected = input − explained missing must equal the output count."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dexengine.data import rules as R
from dexengine.retarget.reretarget_stage import MISMATCH, canon_set, ls_visible, read_lines

EXC_RE = re.compile(
    r"(ValueError|RuntimeError|KeyError|AssertionError|TypeError|IndexError|OSError|FileNotFoundError):"
    r" .*"
)


def outputs(root: Path, ds: str, spec: str, emb: str) -> list[str]:
    """Clip folder of every ``<emb>/<clip>/<x>/trajectory_kinematic.npz`` whose path has ``/0/`` (links not
    followed)."""
    base = root / "processed" / ds / spec / emb
    out = []
    try:
        clips = list(os.scandir(base))
    except OSError:
        return out
    for c in clips:
        if not c.is_dir(follow_symlinks=False):
            continue
        for x in os.scandir(c.path):
            f = os.path.join(x.path, "trajectory_kinematic.npz")
            if x.is_dir(follow_symlinks=False) and os.path.lexists(f) and "/0/" in f:
                out.append(c.name)
    return out


def count_out(root: Path, ds: str, spec: str, emb: str, scope_cur: Path | None) -> int:
    """Done clips (with a scope: those in scope only)."""
    done = outputs(root, ds, spec, emb)
    return len(canon_set(done, ds) & canon_set(read_lines(scope_cur), ds)) if scope_cur else len(done)


def in_list(in_dir: Path, scope_cur: Path | None) -> list[str]:
    return read_lines(scope_cur) if scope_cur else ls_visible(in_dir)


def missing_out(in_dir: Path, root: Path, ds: str, spec: str, emb: str, scope_cur: Path | None) -> list[str]:
    """Input clips without output — a set difference of the two name streams after the name rule (never log
    scraping: eight interleaved workers do not keep a clip's lines together)."""
    return sorted(canon_set(in_list(in_dir, scope_cur), ds) - canon_set(outputs(root, ds, spec, emb), ds))


def report_missing(missing: list[str], ik_log: Path) -> None:
    """The clips without output and the exception lines of the ik log, as they are (no guessing at reasons)."""
    if not missing:
        return
    print(f"[reretarget] {len(missing)} clips without output:")
    for c in missing:
        print(f"    - {c}")
    print("[reretarget] exceptions in the log (shared by every phase):")
    try:
        exc = sorted({m.group(0) for m in EXC_RE.finditer(ik_log.read_text(encoding="utf-8", errors="replace"))})
    except OSError:
        exc = []
    for e in exc[:3]:
        print(f"    {e}")
    sys.stdout.flush()


def count_check(in_dir: Path, root: Path, ds: str, emb: str, label: str, ctx, scope_cur: Path | None) -> bool:
    """The count rule (module docstring). An exclusion listed for a clip that has output does not lower the
    expectation, so a stale exclusion cannot hide a real mismatch. Returns False on failure."""
    n_in = len(in_list(in_dir, scope_cur))
    n_out = count_out(root, ds, ctx.spec, emb, scope_cur)
    if scope_cur:
        print(
            f"[RERETARGET_SCOPE] {label}: {count_out(root, ds, ctx.spec, emb, None) - n_out} outputs outside the "
            "scope (not counted)"
        )
    miss = set(missing_out(in_dir, root, ds, ctx.spec, emb, scope_cur))
    excl = canon_set(sorted(R.retarget_excluded(ds)), ds)
    expl, unl = sorted(miss & excl), sorted(miss - excl)
    need = n_in - len(expl)
    if expl:
        print(
            f"[reretarget] excluded ds={ds} n={len(expl)} (ledger precompute/{ds}/diag/excluded_clips.txt): "
            + " ".join(expl)
        )
    print(
        f"[reretarget] {label}: input {n_in} − explained {len(expl)} = expected {need} → output {n_out} "
        + ("OK" if need == n_out else MISMATCH),
        flush=True,
    )
    if unl or need != n_out:
        if unl:
            print(
                f"[reretarget] ✗ ds={ds}: {len(unl)} missing clips the ledger does not explain — an exclusion is a "
                "decision and goes into the ledger with its reason",
                file=sys.stderr,
            )
            for c in unl:
                print(f"    - {c}", file=sys.stderr)
        else:
            print(
                f"[reretarget] ✗ ds={ds}: every missing clip is explained (missing {len(miss)}) but the count does "
                f"not match — input {n_in} output {n_out} expected {need}",
                file=sys.stderr,
            )
        sys.stderr.flush()
        report_missing(sorted(miss), ctx.log / f"{ds}_{emb}_ik.log")
        return False
    return True
