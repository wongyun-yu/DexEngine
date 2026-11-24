"""One placement augment run: rolls out the scene variants of one checkpoint (cell) through the frozen judge entry.

It uses the same pieces as record stage 1 (canonical DUMP_ONLY dump -> session), with three differences:
  1. data root = the shadow root (DEXENGINE_MAIN_ROOT · SEQ_DIR), where the variant clips are (augment.place_gen)
  2. list = the variant names · NE = variants x epc · evaluations per clip epc (run_spec argument; canonical
     policy.EPC)
  3. session = python -m dexengine.record.augment.session (AugRecorder, via session.launch.isaac_cmd)
Checkpoint: variants are clips outside the training list, so they cannot get dexengine.rl.checkpoints's generation check
(in-cell checkpoint); an out-of-cell copy (shadow root _ckpt/, sha256 = the original) takes the 'out-of-cell checkpoint'
path dexengine.rl.checkpoints decides (marker [CANON_USE] …).
Entry = the record augment step (augment.stage rollout: cell_spec + run_spec, mode A only; no command line entry in
this file).
Outputs <out>/{clips.txt, variants.json, dryrun.out, session.log, rollouts.npz, diag.json, run.json}
Markers: [AUG_RUN] … · end line [AUG_RUN] DONE ok=…
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.hashing import sha256_file
from dexengine.core.repo import REPO_ROOT
from dexengine.record.augment.run_session import run_session

HERE = Path(__file__).resolve().parent
WT = REPO_ROOT  # tree root
import hashlib
import types

from dexengine.record.session import launch as CLA  # noqa: E402  (owner of the progress watch: watch_stall · STALL_S)

MARK = "[AUG_RUN]"
SESSION_TIMEOUT_S = 5400
DONE_LINE = "[TRAJ_REC] DONE"
CLOSE_HANG = "[AUG_CLOSE_HANG]"
#: rc of a session cut during closing: timeout(1) expiry · TERM/KILL from the watch (Popen negative · shell 128+n)
_CUT_RC = (CLA.TIMEOUT_RC, -15, -9, 137, 143)


def _bundle():
    """record.bundle, imported lazily (it imports rl, which loads site modules at import time)."""
    from dexengine.record import bundle

    return bundle


def session_ok(rc: int, text: str, n_variants: int, outputs_ok: bool, places: int | None = None) -> tuple[bool, str]:
    """Verdict of one augment session: (ok, branch), branch ∈ ok · close_skip · close_hang · stall · fail (reason
    after it). Single owner of 'DONE present -> ok' (the session caller augment.run and readers of old pod logs read
    it alike). ok depends on outputs: [TRAJ_REC] DONE (count equation) · [AUG_PLACE] = variant count · trajectory and
    diag files (outputs_ok). rc 0 is ok (close_skip with the close-skip marker). With rc != 0 only a cut after the
    closing-start marker (printed only after DONE and the output check) (_CUT_RC or the watch's stall line) is ok as
    close_hang; a stall before the marker (boot, no progress) is stall, anything else fail."""
    from dexengine.record.augment import session as AS  # noqa: PLC0415  (home of the marker names, session side)

    # log-tail readers pass the AUG_RUN end line's count
    places = text.count("[AUG_PLACE]") if places is None else places
    from dexengine.rl import isaac_cli as KC  # noqa: PLC0415  (home of the closing-start marker, stdlib only)

    done, closing = DONE_LINE in text, KC.CLOSING_MARK in text
    stalled = CLA.STALL_LINE in text
    base = done and places == n_variants and outputs_ok
    why = (
        f"rc {rc} · DONE {'present' if done else 'missing'} · [AUG_PLACE] {places}/{n_variants} · "
        f"outputs {'present' if outputs_ok else 'missing'}"
    )
    if rc == 0:
        if base:
            return True, ("close_skip" if AS.CLOSE_SKIP in text else "ok") + " · " + why
        return False, "fail · " + why
    if closing and (stalled or rc in _CUT_RC):
        if base:
            return True, "close_hang · " + why + " · cut after the closing-start marker"
        return False, "fail · " + why + " · cut after the closing-start marker but outputs do not match"
    if stalled:
        return False, (
            "stall · " + why + " · stalled before the closing-start marker (boot or no progress; hand out again)"
        )
    return False, "fail · " + why


_RUN_END = re.compile(r"\[AUG_RUN\] DONE ok=\w+ rc=(-?\d+) session [\d.]+ s · \[AUG_PLACE\] (\d+)/(\d+)")


def pod_log_verdict(bundle_text: str, session_text: str) -> tuple[bool, str]:  # noqa: V103 — tests only
    """session_ok entry for readers that have only logs (logs/fail of an earlier pod, a tail per file); session_ok
    alone decides. rc and place marks a/b = the last `[AUG_RUN] DONE … rc=N … [AUG_PLACE] a/b` line of the bundle log
    (counted by augment.run over the whole session log) · text = one session log (run__session.log) · outputs = two
    log witnesses (`[TRAJ_REC] DONE … file ` (printed by rl.eval.traj_record only after the atomic write) and the
    session's diag line session.DIAG_WRITTEN (printed after writing the diag)). Without the AUG_RUN end line it dies
    (no verdict without knowing what ran)."""
    m = list(_RUN_END.finditer(bundle_text))
    if not m:
        raise InputMissingError(f"{MARK} no '[AUG_RUN] DONE … rc= … [AUG_PLACE] a/b' line in the pod log: no verdict")
    rc, a, b = (int(x) for x in m[-1].groups())
    lines = session_text.splitlines()
    done = [ln for ln in lines if ln.startswith(DONE_LINE)]
    from dexengine.record.augment import session as AS  # noqa: PLC0415

    diag = any(ln.startswith(AS.DIAG_MARK) and AS.DIAG_WRITTEN in ln for ln in lines)
    outputs = bool(done) and " file " in done[-1] and diag  # " file " = the file word of the rl TRAJ_REC DONE line
    return session_ok(rc, session_text, b, outputs, places=a)


# ledgers beside the checkpoint (run folder): they move with the checkpoint copy so the frozen entry's check reads the
# training conditions. Name owners: collision preset = rl.launch.records · recipe = rl.launch.run_dir.LEDGER_NAME
# · extra slots = rl.env_setup.build EXTRA_SLOTS_LEDGER.
RUN_LEDGERS = ("coll_preset.txt", "hand_recipe.txt", "extra_slots.txt")


def load_variants(a) -> tuple[dict, list]:
    want = set(a.status.split(","))
    V, base_clips, metas = {}, [], []
    for vp in a.variants:
        m = json.loads(Path(vp).read_text())
        metas.append(m)
        base_clips.append(m["clip"])
        vs = [v for v in m["variants"] if v["status"] in want]
        if a.max_variants:
            vs = vs[: a.max_variants]
        for v in vs:
            V[v["name"]] = {
                "dx": v["dx"],
                "dy": v["dy"],
                "yaw": v["yaw"],
                "r": v["r"],
                "clip": m["clip"],
                "k": v["k"],
                "pivot_tree": v.get("pivot_tree"),
            }
    if not V:
        raise SystemExit(f"{MARK} 0 variants to use")
    hands = {m["hand"] for m in metas}
    if a.hand and hands != {a.hand}:
        raise SystemExit(f"{MARK} variant hands {hands} != --hand {a.hand}")
    return V, base_clips


def copy_replace(src: Path, dst: Path) -> None:
    """Write to a shared shadow place (checkpoint copy, run ledgers): skip when the content is the same; otherwise write
    a unique temp file in the same folder and os.replace (atomic). Even when sessions of the same shadow and cell start
    together, readers see only the whole old or the whole new file (overwriting with shutil.copy2 let another session
    read an empty coll_preset.txt between truncate and write)."""
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and dst.read_bytes() == src.read_bytes():
        return
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{time.time_ns()}.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def cell_spec(cell: dict, hand: str, main: Path, shadow: Path, base_clips: list) -> dict:
    """Resolved cell (record cell_of form) -> spec (without variants); the checkpoint copy goes to the shadow _ckpt
    (the record augment step uses this too)."""
    seq_main = next(t.split("=", 1)[1] for t in cell["env"] if t.startswith("SEQ_DIR="))
    if not seq_main.startswith(str(main) + "/"):
        raise SystemExit(f"{MARK} SEQ_DIR {seq_main} is not under the data root {main}")
    run_dir = Path(cell["ckpt"]).parent.parent
    ck_rel = (
        Path("_ckpt")
        / hand
        / cell["campaign"]
        / cell["bundle"]
        / run_dir.name
        / "checkpoints"
        / Path(cell["ckpt"]).name
    )
    ck = shadow / ck_rel
    ck.parent.mkdir(parents=True, exist_ok=True)
    if not ck.is_file() or sha256_file(ck) != cell["ckpt_sha"]:
        copy_replace(Path(cell["ckpt"]), ck)
    for fn in RUN_LEDGERS:
        if (run_dir / fn).is_file():
            copy_replace(run_dir / fn, ck.parent.parent / fn)
    if sha256_file(ck) != cell["ckpt_sha"]:
        raise SystemExit(f"{MARK} checkpoint copy sha256 mismatch")
    env = [t for t in cell["env"] if not t.startswith(("SEQ_DIR=", "DEXENGINE_SEQ_LIST="))]
    return {
        "hand": hand,
        "campaign": cell["campaign"],
        "bundle": cell["bundle"],
        "ckpt_orig": str(cell["ckpt"]),
        "ckpt_rel": str(ck_rel),
        "ckpt_sha256": cell["ckpt_sha"],
        "iter": cell["iter"],
        "env": env,
        "seq_rel": seq_main[len(str(main)) + 1 :],
        "base_clips": base_clips,
        "run_files": [
            str(Path(ck_rel).parent.parent / fn)
            for fn in RUN_LEDGERS
            if (shadow / Path(ck_rel).parent.parent / fn).is_file()
        ],
    }


def _run_session(tok, tag, out, launch_args, run, shadow, gpu, mode, k_keep, names):
    """Dry run, the record session, its verdict and run.json."""
    t0 = time.time()
    _bundle().dryrun(WT, tok, tag, out / "dryrun.out", launch_args)
    run["dryrun_s"] = round(time.time() - t0, 1)
    from dexengine.core.paths import run_root_of  # the launch's dry log folder: its run tree (Request.run_tree)
    from dexengine.rl.launch import records as LR

    dump = LR.log_dir(True, run_root_of(shadow, WT)) / f"{tag}_frozenreplay_train.log"
    senv = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        AUG_VARIANTS=str(out / "variants.json"),
        AUG_MODE=mode,
        AUG_DIAG=str(out / "diag.json"),
    )
    if k_keep is not None:
        senv["AUG_K_KEEP"] = str(int(k_keep))
    t0 = time.time()
    rc = run_session(
        [
            "timeout",
            "-k",
            "60",
            str(SESSION_TIMEOUT_S),
            *CLA.isaac_cmd("dexengine.record.augment.session", ["--dump", str(dump)]),
        ],
        WT,
        senv,
        out / "session.log",
        info=run,
    )
    run.update(rc=rc, session_s=round(time.time() - t0, 1), t_end=time.time())
    txt = (out / "session.log").read_text(errors="replace")
    run["done_mark"] = DONE_LINE in txt
    run["aug_place_marks"] = txt.count("[AUG_PLACE]")
    from dexengine.rl import checkpoints as _CK  # noqa: PLC0415  (owner of the out-of-cell marker)

    run["canon_use_outside"] = f"{_CK.TAG} {_CK.OUTSIDE_CELL}" in txt
    ok, kind = session_ok(rc, txt, len(names), (out / "rollouts.npz").is_file() and (out / "diag.json").is_file())
    run["verdict"] = kind
    if ok:
        _bundle().TK.require_mark(out / "session.log")  # proof the bare kit really applied
    (out / "run.json").write_text(json.dumps(run, indent=1))
    if ok and kind.startswith("close_hang"):
        print(
            f"{CLOSE_HANG} session did not end after the closing-start marker and was cut: ok by the output check · "
            f"{kind}",
            flush=True,
        )
    print(
        f"{MARK} DONE ok={ok} rc={rc} session {run['session_s']} s · [AUG_PLACE] {run['aug_place_marks']}/"
        f"{len(names)} · verdict {kind.split(' · ')[0]} · {out}",
        flush=True,
    )
    return 0 if ok else 1


def _stage_spec(out, spec, shadow):
    """Write variants.json and clips.txt; the shadow trees and the checkpoint must match the spec."""
    from dexengine.core.paths import TREE_SFX

    out.mkdir(parents=True, exist_ok=True)
    V = spec["variants"]
    names = sorted(V)
    (out / "variants.json").write_text(json.dumps(V, indent=1))
    (out / "clips.txt").write_text("".join(n.removesuffix(TREE_SFX) + "\n" for n in names))
    seq = shadow / spec["seq_rel"]
    if not seq.is_dir():
        raise InputMissingError(f"{MARK} shadow SEQ_DIR missing {seq}")
    miss = [n for n in names if not (seq / n / "0/trajectory_kinematic.npz").is_file()]
    if miss:
        raise SystemExit(f"{MARK} {len(miss)} variants missing from the shadow: {miss[:3]}")
    ck = shadow / spec["ckpt_rel"]
    if sha256_file(ck) != spec["ckpt_sha256"]:
        raise SystemExit(f"{MARK} checkpoint sha256 != spec {ck}")
    return names, seq, ck


def run_spec(
    spec: dict,
    shadow: Path,
    out: Path,
    mode: str,
    epc: int,
    gpu: str,
    source: str,
    k_keep: int | None = None,
    launch_args=(),
) -> int:
    """k_keep = cap of successful rollouts kept per variant (canonical record augment = policy.K_KEEP given explicitly
    -> session env AUG_K_KEEP · None = the caller's env as it is). launch_args = launcher arguments after the run name
    (RL switches as `--set field=value`). mode is A (policy rollout) only, source only empty;
    anything else dies (the pilot B and Aeq branches are gone; both arguments remain as positional arguments of the
    caller augment.stage rollout)."""
    if mode != "A" or source:
        raise InputMissingError(f"{MARK} mode {mode!r} · source {source!r}: mode A only, no source")

    names, seq, ck = _stage_spec(out, spec, shadow)
    queue = os.environ.get("DEXENGINE_RL_QUEUE_ROOT", "") or str(out)
    ne = epc * len(names)
    tag = f"aug{mode}_{os.getpid()}_{int(time.time()) % 10**6}"
    tok = _bundle().env_set(
        spec["env"],
        SEQ_DIR=seq,
        DEXENGINE_SEQ_LIST=out / "clips.txt",
        GPU=gpu,
        DEXENGINE_MAIN_ROOT=shadow,
        DEXENGINE_RL_QUEUE_ROOT=queue,
        FROZEN_REPLAY_CKPT=ck,
        FROZEN_REPLAY_NE=ne,
        FROZEN_REPLAY_EVALS_PER_CLIP=epc,
        FROZEN_REPLAY_ITER=spec["iter"],
        DEXENGINE_TRAJ_RECORD=out / "rollouts.npz",
    )
    from dexengine.rl import cell_session as CR  # lazy: rl imports site modules at load

    staged = CR.stage_coll_preset(ck.parent.parent, WT, tag, data_root=shadow)
    git = lambda *c: subprocess.run(["git", "-C", str(WT), *c], capture_output=True, text=True).stdout.strip()  # noqa: E731
    run = {
        "mode": mode,
        "hand": spec["hand"],
        "campaign": spec["campaign"],
        "bundle": spec["bundle"],
        "ckpt": spec["ckpt_orig"],
        "ckpt_copy": str(ck),
        "ckpt_sha256": spec["ckpt_sha256"],
        "iter": spec["iter"],
        "n_variants": len(names),
        "epc": epc,
        "ne": ne,
        "base_clips": spec["base_clips"],
        "tree": str(WT),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "host": os.uname().nodename,
        "shadow": str(shadow),
        "gpu": gpu,
        "coll_preset_staged": bool(staged),
        "k_keep": k_keep,
        "t_start": time.time(),
    }
    print(
        f"{MARK} {mode} · {spec['hand']} · {spec['campaign']}/{spec['bundle']} · "
        f"checkpoint {ck.name}@{spec['iter']} · variants {len(names)} x {epc} = NE {ne} · host={run['host']} "
        f"tree={WT} "
        f"commit={run['commit'][:9]} branch={run['branch']}",
        flush=True,
    )
    return _run_session(tok, tag, out, launch_args, run, shadow, gpu, mode, k_keep, names)


CELL_BATCH_MARK = "[AUG_CELL_BATCH]"


def rollable(m: dict, problem, tech) -> bool:
    """Is this a generation record that one_clip takes to rollout? Not for a canonical cell technical failure (job
    failure, one_clip kills it), a dropped canonical cell or 0 passing variants (that clip moves on without rollout)."""
    return not tech(m) and not problem(m) and any(v["status"] == "ok" for v in m["variants"])


def link_run(gd: Path, rd: Path) -> None:
    """Make the clip work place rd (= <d>/run) a real directory with a hard link per file of the bundle run gd: the
    piece carrying walk (clean.classify; os.walk does not follow linked directories) carries the clip place run/*, and
    tar carries one inode as link entries (the receiving tar_index counts by the target's size and sha). Another file
    system (EXDEV) dies; no fallback to copying. Everything is linked under the temp place <rd>.tmp, then renamed (a
    half run makes the next run drop that clip from the bundle and read the half) · symlinks inside gd (the walk does
    not follow them; silently missing) die."""
    if links := [str(Path(dp) / x) for dp, dn, fs in os.walk(gd) for x in dn + fs if (Path(dp) / x).is_symlink()]:
        raise SystemExit(f"{CELL_BATCH_MARK} ✗ symlinks {links[:3]} inside the bundle run {gd}: cannot hard-link them")
    tmp = rd.with_name(rd.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for dp, _dn, fs in os.walk(gd):
        for f in fs:
            src = Path(dp) / f
            dst = tmp / src.relative_to(gd)
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError as e:
                raise SystemExit(
                    f"{CELL_BATCH_MARK} ✗ hard link {src} -> {dst} failed ({e}): no fallback to copying"
                ) from e
    tmp.rename(rd)


def run_cell(a, cell: dict, clips: list[str], gens: list[tuple], shadow: Path, problem, tech) -> int:
    """gens = per clip (bundle, work place d, generation record vj) · problem/tech = augment.stage canon_problem /
    canon_tech_fail -> clips batched (0 = not batched: fewer than 2 clips to roll)."""
    rows = [
        (c, d, vj)
        for c, (_b, d, vj) in zip(clips, gens, strict=True)
        if rollable(json.loads(Path(vj).read_text()), problem, tech) and not (Path(d) / "run").exists()
    ]
    if len(rows) < 2:
        return 0
    from dexengine.record.augment import policy as AP

    V, base = load_variants(
        types.SimpleNamespace(status="ok", variants=[str(vj) for _c, _d, vj in rows], max_variants=0, hand=a.hand)
    )
    spec = dict(cell_spec(cell, a.hand, a.main, shadow, base), variants=V)
    tag = hashlib.sha256("\n".join(sorted(c for c, _d, _v in rows)).encode()).hexdigest()[:12]
    gd = Path(rows[0][1]).parent / f"_cell_{tag}" / "run"
    if len({Path(d).parent for _c, d, _v in rows}) != 1:
        raise SystemExit(
            f"{CELL_BATCH_MARK} ✗ cell {cell['bundle']}: the clip work places have different judge "
            "fingerprints (one cell = one fingerprint)"
        )
    t0 = time.time()
    rj = gd / "run.json"
    if not (rj.is_file() and json.loads(rj.read_text()).get("rc") == 0 and (gd / "rollouts.npz").is_file()):
        if run_spec(spec, shadow, gd, "A", AP.EPC, a.gpu, "", k_keep=AP.K_KEEP) != 0:
            raise SystemExit(f"{CELL_BATCH_MARK} ✗ cell {cell['bundle']} bundle rollout failed: {gd}/session.log")
    for _c, d, _vj in rows:
        link_run(gd, Path(d) / "run")
    print(
        f"{CELL_BATCH_MARK} cell {cell['bundle']} · clips {len(rows)}/{len(clips)} · variants {len(V)} x {AP.EPC} = "
        f"NE {len(V) * AP.EPC} · session {time.time() - t0:.0f} s · {gd}",
        flush=True,
    )
    return len(rows)
