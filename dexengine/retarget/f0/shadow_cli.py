"""f0 prejudge run helpers: the shadow-root lock, shadow seeding, the shadow root and the upstream completeness report
(used by prejudge.main; the argument parser stays in prejudge.py, the step script)."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from dexengine.core.paths import f0_root
from dexengine.retarget.f0.shadow_key import clip_sides
from dexengine.retarget.f0.shadow_status import SIDES

# When two prejudges solve the same shadow root (<f0 root>/<hand spec>/, where LEDGER.tsv lives), one side's mirroring
# (mirror_clip: delete and rebuild the shadow) overwrites a clip the other is solving; that side's worker writes the
# repair ledger into the new folder, and this side's worker ends with the ALREADY_REPAIRED line and no marker. Then
# FAIL_NO_BUDGET_MARK becomes the last line of the shared ledger and even successful chunks read as failed (measured
# when a full stage-1 f0 run overlapped chunk preparation). So writing runs take turns per root through one lock file
# (flock, released when the process ends). Waiting longer than F0_ROOT_LOCK_WAIT_S is fatal. The output does not
# depend on the order (the later run is usually a cache hit). Not holding a GPU while waiting is the launcher's job.
F0_ROOT_LOCK_WAIT_S = 7200


F0_ROOT_LOCK_NAME = ".f0_prejudge.lock"


def shadow_root_lock(root: Path, run_id: str):
    """Lock the shadow root -> the locked file object (held until the process ends; closing releases it).
    Markers: one `[F0_ROOT_LOCK] acquired waited=…s root=…` line when acquired, `[F0_ROOT_LOCK] waiting …` while another
    process holds it (one elapsed line every 60 s). Beyond the limit it dies with `[F0_ROOT_LOCK] ✗`."""
    import fcntl  # noqa: PLC0415

    fh = open(root / F0_ROOT_LOCK_NAME, "a+")  # noqa: SIM115 — held until the process ends

    def _try() -> bool:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _holder() -> str:
        fh.seek(0)
        return fh.read().strip().replace("\n", " ")[:160] or "?"

    t0 = time.time()
    if not _try():
        holder = _holder()
        print(
            f"[F0_ROOT_LOCK] waiting — another prejudge ({holder}) holds the same shadow root root={root}", flush=True
        )
        last = t0
        while not _try():
            if time.time() - t0 > F0_ROOT_LOCK_WAIT_S:
                raise SystemExit(f"[F0_ROOT_LOCK] ✗ not acquired after {F0_ROOT_LOCK_WAIT_S}s ({holder}) root={root}")
            if time.time() - last > 60:
                print(f"[F0_ROOT_LOCK] still waiting {time.time() - t0:.0f}s ({holder})", flush=True)
                last = time.time()
            time.sleep(2)
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} run={run_id} t={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    fh.flush()
    print(f"[F0_ROOT_LOCK] acquired waited={time.time() - t0:.0f}s root={root}", flush=True)
    return fh


def seed_shadows(src_root: Path, dst_root: Path, sp: str, emb: str, clips) -> dict:
    """f0 cache seed for an isolated run: copies these clips' shadows from the canonical shadow root
    (<src>/<hand spec>/<emb>/<clip>/) and their wrist shadows (<src>/<hand spec>/wrist_poses_<side>/<clip>.npz) into the
    isolated shadow root as real copies. Never hard links or symlinks: the moment the isolated run writes in place, the
    canonical copy would change too. Existing entries are kept (idempotent); the ledger (LEDGER.tsv) and summaries are
    not copied (the isolated run's f0 writes its own). Returns counts (copied, have, no_src, wrist)."""
    src, dst = Path(src_root) / sp, Path(dst_root) / sp
    if src.resolve() == dst.resolve():
        raise SystemExit(f"[F0_SEED] ✗ seed source and destination are the same: {src}")
    n = {"copied": 0, "have": 0, "no_src": 0, "wrist": 0}
    (dst / emb).mkdir(parents=True, exist_ok=True)
    for c in clips:
        s, d = src / emb / c, dst / emb / c
        if d.exists():
            n["have"] += 1
        elif s.is_dir():
            tmp = d.with_name(d.name + ".seed_tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(s, tmp, symlinks=False)  # real copy (follows links, copies content)
            os.replace(tmp, d)
            n["copied"] += 1
        else:
            n["no_src"] += 1
        for sd in SIDES:
            ws, wd = src / f"wrist_poses_{sd}" / f"{c}.npz", dst / f"wrist_poses_{sd}" / f"{c}.npz"
            if ws.is_file() and not wd.exists():
                wd.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ws, wd)
                n["wrist"] += 1
    return n


def _seed_only(a, clips, shadow_root: Path, sp: str) -> int:
    """`--seed-from` mode: only seed and print one `[F0_SEED]` marker line. Never seeds the default root (--f0-root is
    required)."""
    if not a.f0_root:
        raise SystemExit(
            "[F0_SEED] ✗ --seed-from works only with --f0-root (the isolated shadow root) — "
            "the default root is never seeded"
        )
    n = seed_shadows(Path(a.seed_from), shadow_root, sp, a.emb, clips)
    print(
        f"[F0_SEED] shadows {n['copied']}/{len(clips)} copied · already present {n['have']} · "
        f"not in canonical {n['no_src']} · "
        f"wrist {n['wrist']} (from={Path(a.seed_from) / sp} to={shadow_root / sp})",
        flush=True,
    )
    return 0


def resolve_shadow_root(dataset: str, f0_root_arg: str) -> Path:
    """Shadow root (<here>/<hand spec>/<emb>/<clip>). The owner is paths.f0_root(dataset) (env
    DEXENGINE_F0_STAGE wins); --f0-root gives that location directly — one place assembles the path."""
    if f0_root_arg:
        return Path(f0_root_arg)
    return f0_root(dataset)


def _report_upstream_completeness(clips, stage_src, wrist_dirs, a):
    """Check within ~10 s that wrist caches exist and are fresh. Fresh = the cache equals the values extracted from the
    current staging body_positions (owner: placement.fk.wrist_poses.cache_state). Any differing cache is fatal, flag or
    not: f0 carries that wrist into pads and shadows and the judge uses the shadow as the arm target (a stale cache
    would let judge and f0 use the old wrist while the bake uses the new one)."""
    from dexengine.placement.fk import wrist_poses as WP  # noqa: PLC0415 — owner of wrist-cache freshness

    _missing, _stale, _fresh = {}, {}, 0
    for _c in clips:
        try:
            _cs = clip_sides(stage_src, _c)
        except Exception:  # noqa: BLE001 — any unreadable tree counts as two-handed here; it fails later on its own
            _cs = SIDES  # with no tree it dies later as 'staging missing' — here it is counted as two-handed
        _bp = WP.find_body_positions_file(str(stage_src / _c))
        for _side in _cs:
            _w = wrist_dirs[_side] / f"{_c}.npz"
            if not _w.exists():
                _missing.setdefault(_c, []).append(_side)
                continue
            if _bp is None:  # staging missing; that clip dies later with a per-clip reason
                continue
            try:
                _why = WP.cache_state(_bp, _w, _side)[0]
            except Exception as e:  # noqa: BLE001 — freshness cannot be proven without extraction
                _why = f"extraction failed({type(e).__name__}: {str(e)[:80]})"
            if _why:
                _stale.setdefault(_c, []).append(f"{_side}:{_why}")
            else:
                _fresh += 1
    if _stale:
        print(
            f"[WRIST_CACHE] ✗ f0 upstream check: wrist caches differ from the current body_positions "
            f"{len(_stale)}/{len(clips)} clips",
            flush=True,
        )
        for _c, _sd in sorted(_stale.items())[:20]:
            print(f"    {_c}  [{', '.join(_sd)}]", flush=True)
        raise SystemExit(
            f"[f0_prejudge] stale wrist caches in {len(_stale)} clips — run stage 6 first "
            "(argv: data/stages.py --emit-run 6; --skip_fresh rewrites only the stale ones)"
        )
    print(
        f"[WRIST_CACHE] f0 upstream check fresh={_fresh} stale=0 missing={sum(map(len, _missing.values()))}", flush=True
    )
    if _missing:
        # Note: with no tree the count above is **two-handed**, so a single-hand emb clip
        # is caught here because of the side it does not have — this banner is not "why did this clip
        # fail". The real reason is in the per-clip lines below (e.g. `no manipulating hand … D23
        # inactive → no_main_object`).
        print(
            f"[f0_prejudge] ★ upstream missing {len(_missing)}/{len(clips)} clips — no wrist cache"
            " (single-hand embs can be caught by their absent side — see the per-clip lines for the real reason)",
            flush=True,
        )
        for _c, _sd in sorted(_missing.items())[:20]:
            print(
                f"    {_c}  [{','.join(_sd)}]  ← {wrist_dirs['right'].parent}/<wrist_poses_side>/{_c}.npz", flush=True
            )
        if len(_missing) > 20:
            print(f"    … and {len(_missing) - 20} more", flush=True)
        if getattr(a, "require_complete", False):
            raise SystemExit("[f0_prejudge] --require-complete: upstream inputs missing, stopping")
    else:
        print(f"[f0_prejudge] upstream complete OK — wrist caches {len(clips)}/{len(clips)}", flush=True)
