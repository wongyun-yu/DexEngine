#!/usr/bin/env python3
"""Parts of the public-data record driver (record.driver): queue, bundle output checks, machine state (GPU, memory,
tree). The rules are stated in the record.driver module doc (this file holds their functions; no caller outside the
driver)."""

from __future__ import annotations

import collections
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from dexengine.core import paths as P

# base exclusion reason rule (base_exclusions) · augment failure marker (AUG_FAIL)
from dexengine.record import chain as CANON
from dexengine.record.augment import stage as AUGC  # owner of the augment marker (MARK)
from dexengine.record.looks import look_d as LOOK  # record render look (D): episode meta check
from dexengine.record.render import writer as CD  # depth file format (frame count, header only)
from dexengine.record.store import store as ST  # video frame count (_ffprobe) · key names
from dexengine.record.store.tasks import episode_file  # episode file place (info.json form, chunk); owner store.tasks

MARK = "[COLLECT_PILOT]"


def _acceptance():
    """rl.acceptance (ledger reader), imported lazily: rl loads plugin modules at import time."""
    from dexengine.rl import acceptance

    return acceptance


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def die(msg: str) -> None:
    raise SystemExit(f"{MARK} ✗ {msg}")


def say(msg: str) -> None:
    print(f"{MARK} {now()} {msg}", flush=True)


def write_atomic(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, p)


def tree_state(tree: Path) -> str:
    """Condition = the code tree: HEAD + the list of changes in code paths. When it changes during a run no new bundle
    starts."""
    head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True)
    st = subprocess.run(
        ["git", "-C", str(tree), "status", "--porcelain", "--", "scripts", "source", "pipeline"],
        capture_output=True,
        text=True,
    )
    if head.returncode or st.returncode:
        return f"git failed rc {head.returncode}/{st.returncode}"
    return head.stdout.strip() + "|" + hashlib.sha256(st.stdout.encode()).hexdigest()[:12]


def mem_available_gb() -> float:
    m = re.search(r"^MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text(), re.M)
    if not m:
        die("/proc/meminfo has no MemAvailable")
    return int(m.group(1)) / 1e6


def ancestors(pid: int) -> set[int]:
    out, p = set(), pid
    while p > 1 and p not in out:
        out.add(p)
        try:
            m = re.search(r"^PPid:\s+(\d+)", Path(f"/proc/{p}/status").read_text(), re.M)
        except OSError:
            break
        if not m:
            break
        p = int(m.group(1))
    return out


def gpu_foreign(gpu: str, me: int) -> list[int]:
    """Compute processes on the GPU that are not descendants of this driver (nvidia-smi --query-compute-apps). A failed
    query dies."""
    idx = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], capture_output=True, text=True
    )
    if idx.returncode or apps.returncode:
        die(f"nvidia-smi failed rc {idx.returncode}/{apps.returncode}")
    uuid = {i.strip(): u.strip() for i, u in (ln.split(",") for ln in idx.stdout.splitlines() if ln.strip())}
    if gpu not in uuid:
        die(f"GPU {gpu} is not in the nvidia-smi list {sorted(uuid)}")
    pids = [
        int(p) for u, p in (ln.split(",") for ln in apps.stdout.splitlines() if ln.strip()) if u.strip() == uuid[gpu]
    ]
    return [p for p in pids if me not in ancestors(p)]


def adopted(hand: str) -> set[tuple[str, str]]:
    """(axis, clip) pairs the current selection adopts: one reader of the selection, rl.manifest.sources (the same
    read as the training entry and the record readiness oracle). A ledger canon outside the adoption is
    not collected: collecting an old re-split piece (its replacement not yet trained; clip_deselect 'open_twin') would
    add the same motion twice once the replacement is trained (regeneration = a new generation; changes are not carried
    over). When the replacement is trained it enters as adopted ∧ canon."""
    from dexengine.core import paths as P
    from dexengine.data import rules as RR
    from dexengine.placement import train_ready_util as TRU

    files = RR.ledger_files("*", hand, root=P.main_root())
    if not files:
        raise SystemExit(f"[manifest] {hand}: no folded ledger — the stage-1 and judge chain builds it first")
    return {(RR.ledger_axis(f), c) for f in files for c in TRU._adopted(str(f))}


def bundle_queue(
    ledger: Path, order: list[str], min_rate: float, adopt: set[tuple[str, str]], subset: Path | None = None
) -> list[dict]:
    """Ledger -> [{dataset, campaign, bundle, clips}] (canon ∧ adopted by the current selection (adopt = adopted(hand);
    explicit input, no default) ∧ completion rate >= threshold, the same comparison as bundle.pick_clips); with subset
    (a union-first cover list) reduced by subset_queue. Canon outside the adoption is left out with counts per dataset
    printed (even 0; marker [COLLECT_ADOPT]); canon of an axis missing from the selection dies."""
    groups: dict[tuple, list[str]] = collections.defaultdict(list)
    axes = {ax for ax, _c in adopt}
    off: collections.Counter = collections.Counter()
    for r in _acceptance().read_ledger(ledger):
        if r["state"] != "canon":  # pending, awaiting retraining, dropped: the completion field is '-'
            continue
        if r["axis"] not in axes:
            die(
                f"ledger canon {r['axis']} / {r['clip']}: the axis is not in the current selection "
                f"(rl.manifest.sources)"
            )
        if (r["axis"], r["clip"]) not in adopt:
            off[r["dataset"]] += 1
            continue
        tot = int(r["total"])
        if tot and int(r["complete"]) / tot >= min_rate:
            groups[(r["dataset"], r["campaign"], r["bundle"])].append(r["clip"])
    print(
        f"[COLLECT_ADOPT] canon outside the adoption {sum(off.values())} left out"
        + (" (" + " · ".join(f"{d} {n}" for d, n in sorted(off.items())) + ")" if off else "")
        + f" · {Path(ledger).name}",
        flush=True,
    )
    unknown = sorted({k[0] for k in groups} - set(order))
    if unknown:
        die(f"ledger datasets {unknown} are not in --order (the order is an explicit input)")
    # operations cell and work piece key = <campaign>__<bundle>
    seen = collections.Counter(k[1:] for k in groups)
    if mixed := sorted(k for k, n in seen.items() if n > 1):
        die(f"bundles mixing datasets {mixed[:3]}: two datasets use the piece key <campaign>__<bundle>")
    q = [
        {"dataset": k[0], "campaign": k[1], "bundle": k[2], "clips": sorted(v)}
        for k, v in sorted(groups.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][2], kv[0][1]))
    ]
    return q if subset is None else subset_queue(q, subset)


def subset_queue(queue: list[dict], f: Path) -> list[dict]:
    """Queue with only the clips of the list (cover list); empty bundles are dropped · list clips outside the queue die
    (when the ledger changed, the list is rebuilt). Marker [COLLECT_PILOT] subset."""
    want = set(P.read_clip_list(str(f), label="--clip-subset", quiet=True))
    have = {c for b in queue for c in b["clips"]}
    if want - have:
        die(
            f"{len(want - have)} clips of --clip-subset are outside the queue (e.g. {sorted(want - have)[:3]}): "
            f"rebuild the list"
        )
    out = [dict(b, clips=[c for c in b["clips"] if c in want]) for b in queue]
    out = [b for b in out if b["clips"]]
    print(
        f"[COLLECT_PILOT] subset {f.name}: clips {len(want)}/{len(have)} · bundles {len(out)}/{len(queue)}", flush=True
    )
    return out


def bdir_name(b: dict) -> str:
    return f"{b['campaign']}__{b['bundle']}"


def frame_problems(ds: Path, eps: list[dict], info: dict) -> list[str]:
    """Per episode 3 videos and depth frames = video_frames ≈ step time (shared by the base and augment datasets and the
    merged final; paths from the info.json form, episode_file)."""
    step_dt = 1.0 / info["fps"]
    vfps = info["collect"]["video_fps"]
    frames_bad = []
    for e in eps:
        i, n, L = e["episode_index"], e["video_frames"], e["length"]
        got = {
            k: ST._ffprobe(episode_file(ds, info, info["video_path"], i, video_key=k))["frames"]
            for k in ST.VIDEO_KEYS.values()
        }
        got[ST.DEPTH_KEY] = CD.frames(episode_file(ds, info, info["collect"]["depth_path"], i))
        t_frames = (L - 1) * step_dt * vfps + 1
        if any(v != n for v in got.values()) or abs(n - t_frames) >= 1.0:
            frames_bad.append(f"ep{i} {e['clip']}: video_frames {n} · measured {got} · step time {t_frames:.2f}")
    return [f"frame mismatch {len(frames_bad)}: {frames_bad[:2]}"] if frames_bad else []


def aug_fail_line(log: str) -> str:
    """The augment failure line of a run log (chain.AUG_FAIL; the last one, '' if none)."""
    return next((ln for ln in reversed(log.splitlines()) if ln.startswith(CANON.AUG_FAIL)), "")


def camera_problems(info: dict, eps: list[dict]) -> list[str]:
    """Are the camera values inside the dataset? Intrinsics info.json collect.cameras · extrinsics per episode
    camera_pose (rules store.meta.camera_intrinsics, camera_pose)."""
    bad = [] if (info.get("collect") or {}).get("cameras") else ["info.json collect.cameras missing"]
    miss = [e["episode_index"] for e in eps if not e.get("camera_pose")]
    return bad + ([f"episodes without camera_pose {len(miss)}: {miss[:3]}"] if miss else [])


def _read_ds(ds: Path) -> tuple[list[dict], dict]:
    eps = [json.loads(ln) for ln in (ds / "meta/episodes.jsonl").read_text().splitlines() if ln.strip()]
    return eps, json.loads((ds / "meta/info.json").read_text())


def aug_count_problems(out: Path, saved: set[str], excl: dict, run_log: str) -> tuple[list[str], dict[str, str]]:
    """Augment check (per-clip bundles; the augment only records): (1) end marker (RECORD DONE) and failure line
    (2) counts.json clip rows = base store · skipped = base exclusions (sets) (3) per row chosen = stored = number of
    picked_names · no duplicate variant names -> (problems, {variant: original clip})."""
    lines = run_log.splitlines()
    bad = []
    done = "RECORD DONE"
    if not any(ln.startswith(f"{AUGC.MARK} {done}") for ln in lines):
        bad.append(f"augment end marker '{AUGC.MARK} {done}' missing: bundle invalid")
    if aug_fail_line(run_log):
        bad.append(f"augment failure line {aug_fail_line(run_log)[:200]}")
    cj = out / "aug" / "counts.json"
    if not cj.is_file():
        return bad + [f"augment count record missing {cj}"], {}
    cnt = json.loads(cj.read_text())
    rows = {r["clip"]: r for r in cnt.get("counts", [])}
    # canonical cell holds (base store, outside the augment, reason in skipped): not a bundle failure
    held = CANON.held_of(run_log)
    if held - saved:
        bad.append(f"held at the canonical cell but not base-stored {sorted(held - saved)[:3]}")
    if set(rows) != saved - held:
        bad.append(
            f"augment clips != base store - held: missing {sorted(saved - held - set(rows))[:3]} · "
            f"extra {sorted(set(rows) - (saved - held))[:3]}"
        )
    sk = set(cnt.get("skipped") or {})
    if sk != set(excl) | held:
        bad.append(
            f"augment skipped != base exclusions ∪ held: missing {sorted((set(excl) | held) - sk)[:3]} · "
            f"extra {sorted(sk - set(excl) - held)[:3]}"
        )
    if held:
        print(
            f"{CANON.HOLD_SUM} count gate: held {len(held)} {sorted(held)[:3]}, outside the augment count · "
            "base store unchanged",
            flush=True,
        )
    by_name: dict[str, str] = {}
    for c, r in rows.items():
        names = r.get("picked_names")
        if names is None or not (len(names) == r.get("picked") == r.get("stored")):
            bad.append(
                f"{c}: chosen {r.get('picked')} · stored {r.get('stored')} · picked_names "
                f"{None if names is None else len(names)} mismatch"
            )
        for n in names or []:
            if n in by_name:
                bad.append(f"variant {n} chosen by two clips ({by_name[n]} · {c})")
            by_name[n] = c
    return bad, by_name


def source_of(e: dict, want) -> str:
    """The episode's original clip: an original episode its own name · a variant selection.source_clip (a clip whose
    original dropped out of the trajectory choice is still 'stored' as variant episodes)."""
    return e["clip"] if e["clip"] in want else (e.get("selection") or {}).get("source_clip") or e["clip"]


def choice_problems(c: str, tc: dict, cands: list[str], eps: list[dict]) -> list[str]:
    """Trajectory choice record check: candidates (measured + unmeasured) = [original + recorded variants] in order ·
    the choice recomputed by the rule = the record · per episode the render-measured ego_view.grasp_visible = the
    value measured at choice time (same fixed camera, same points; an independent measurement)."""
    from dexengine.record import episodes as EPI

    bad = []
    names = [x["name"] for x in tc.get("cands", [])] + list(tc.get("unprobed", []))
    if names != cands:
        bad.append(f"{c}: choice candidates {names[:5]} != original + recorded variants {cands[:5]}")
    try:
        r0, re, _m = EPI.choose_rows(tc.get("cands", []), int(tc.get("n", 0)))
    except (ValueError, TypeError) as ex:
        return bad + [f"{c}: the choice record cannot be recomputed: {ex}"]
    if re != tc.get("chosen") or r0 != tc.get("row0"):
        bad.append(f"{c}: choice record {tc.get('chosen')} · row 0 {tc.get('row0')} != rule {re} · row 0 {r0}")
    vis = {x["name"]: (x["grasp_visible"], x.get("vis_strict")) for x in tc["cands"]}
    off = sorted(
        {
            e["clip"]
            for e in eps
            if ((e.get("ego_view") or {}).get("grasp_visible"), (e.get("ego_view") or {}).get("vis_strict"))
            != vis.get(e["clip"])
        }
    )
    if off:
        bad.append(f"{c}: render-measured grasp visibility != value measured at choice time {off[:3]}")
    return bad


def clip_problems(out: Path, eps: list[dict], saved: set[str], excl: dict, run_log: str | None, src: dict) -> list[str]:
    """Per-clip render (render.clip) dataset check: stored (trajectory, look) set = the union over originals of the plan
    of the round (src.round) (record.episodes.episodes: orig = original · aug = chosen variants · deck rows
    src.episodes_plan.deck) (set difference) · old pieces without a round are a problem · a variant episode's original
    clip = the clip row that chose that name · a leftover old traj-form augment dataset (aug/dataset) is a problem."""
    from dexengine.record import episodes as EPI

    bad, by_name = [], {}
    if run_log is not None and (src.get("switches") or {}).get("aug_placement"):
        # aug round with 0 variants: augment rows exist, no store and no augment skip
        rx = set(src.get("round_excluded") or {})
        bad, by_name = aug_count_problems(out, saved | rx, {c: w for c, w in excl.items() if c not in rx}, run_log)
    if (out / "aug" / "dataset" / "meta/episodes.jsonl").is_file():
        bad.append(f"per-clip bundle has an old traj-form augment dataset {out / 'aug' / 'dataset'}")
    rnd, ep_plan = src.get("round"), src.get("episodes_plan") or {}
    if rnd not in EPI.ROUNDS or not (ep_plan.get("deck") or {}).get("n_rows") or ep_plan.get("round") != rnd:
        return bad + [
            f"round field missing or off (round {rnd!r} · episodes_plan {ep_plan}): only round-form pieces are checked"
        ]
    n_rows = int(ep_plan["deck"]["n_rows"])
    want, of = set(), {}
    for e in eps:
        of.setdefault(source_of(e, saved), []).append(e)
    for c in saved:
        # chosen order = row order of the render plan (r<row>)
        augs = [n for n, o in by_name.items() if o == c]
        tcs = {json.dumps(e.get("traj_choice"), sort_keys=True) for e in of.get(c, [])}
        if len(tcs) != 1:
            bad.append(f"{c}: trajectory choice records differ between episodes ({len(tcs)} kinds)")
            continue
        tc = json.loads(tcs.pop())
        if (tc is not None) != EPI.aug_on(rnd):
            bad.append(f"{c}: round {rnd} but trajectory choice record {'present' if tc is not None else 'missing'}")
            continue
        # trajectory choice (record.episodes.choose): recompute the record by the rule (oracle)
        if tc is not None:
            bad += choice_problems(c, tc, [c] + augs, of[c])
            augs = tc["chosen"]
        try:
            eps_c = (
                EPI.aug_episodes(c, tc.get("row0"), augs, n_rows)
                if EPI.aug_on(rnd)
                else EPI.episodes(rnd, c, [], n_rows)
            )  # aug = row 0 (off-screen original rule) + chosen variants
            # look gate (DEXENGINE_COLLECT_LOOKS: the same rule look_kept the render plan record.episodes.plan filtered
            #   with)
            want |= {(e["traj"], e["look"]) for e in eps_c if EPI.look_kept(e["look"], EPI.looks_mode())}
        except ValueError as ex:
            bad.append(f"{c}: the round {rnd} plan cannot be rebuilt: {ex}")
    if any(e.get("round") != rnd for e in eps):
        bad.append(f"episode round field != round {rnd}: {sorted({str(e.get('round')) for e in eps})}")
    got = {(e["clip"], (e.get("look") or {}).get("name")) for e in eps}
    if got != want:
        bad.append(f"(trajectory, look) != plan: missing {sorted(want - got)[:3]} · extra {sorted(got - want)[:3]}")
    src_bad = [
        e["clip"]
        for e in eps
        if e["clip"] in by_name and (e.get("selection") or {}).get("source_clip") != by_name[e["clip"]]
    ]
    if src_bad:
        bad.append(f"augment episode original clip != chosen clip row {src_bad[:3]}")
    return bad


def verify(out: Path, clips: list[str], ds: Path, run_log: str | None = None) -> dict:
    """Bundle output check: targets = stored ⊔ reasoned exclusions (sets; reason rule chain.base_exclusions) · per
    episode 3 videos and depth frames = video_frames ≈ step time. run_log = that bundle's record run log (driver, new
    record): when given, the camera fields (info cameras, episode camera_pose) and, for augment-on bundles, the augment
    check (clip_problems -> aug_count_problems) are checked too. The redraw (looks.rev) checks old bundles and does not
    pass it (old bundles have no camera fields or augment name records). ds = the dataset checked (driver = that
    bundle's round place piece.round_dir/dataset, paired with the caches and augment records of the bundle place
    out)."""
    ds = Path(ds)
    eps, info = _read_ds(ds)
    src = json.loads((ds / "meta/collect.json").read_text())
    want = {P.tree_name(c) for c in clips}
    pairs = [(e["clip"], (e.get("look") or {}).get("name")) for e in eps]
    clip_mode = any(n is not None for _, n in pairs)  # per-clip render: a look field per episode
    saved = sorted({source_of(e, want) for e in eps} & want) if clip_mode else [c for c, _ in pairs]
    excl = CANON.base_exclusions(src, want)
    bad = []
    if len(pairs) != len(set(pairs)):
        bad.append(f"stored duplicates {len(pairs) - len(set(pairs))}")
    if set(saved) & set(excl):
        bad.append(f"stored ∩ excluded {sorted(set(saved) & set(excl))[:3]}")
    if set(saved) | set(excl) != want:
        bad.append(
            f"targets - (stored ∪ excluded) {sorted(want - set(saved) - set(excl))[:3]} · "
            f"(stored ∪ excluded) - targets {sorted((set(saved) | set(excl)) - want)[:3]}"
        )
    bad += frame_problems(ds, eps, info)
    look_bad = [f"ep{e['episode_index']}:{e.get('render_preset')}" for e in eps if e.get("render_preset") != LOOK.LOOK]
    if look_bad:
        bad.append(f"render look != {LOOK.LOOK} {len(look_bad)}: {look_bad[:3]}")
    aug_n = None
    if clip_mode:
        bad += clip_problems(out, eps, set(saved), {c: w for c, w in excl.items() if c in want}, run_log, src)
        aug_n = sum(1 for c, _ in pairs if c not in want)
    if run_log is not None:
        bad += camera_problems(info, eps)
    loop = collections.Counter(e.get("loophole") or "none" for e in eps)
    return dict(
        ok=not bad,
        problems=bad,
        target=len(want),
        episodes=len(pairs),
        clips=len(saved),
        excluded=excl,
        frames=int(info["total_frames"]),
        video_frames=sum(e["video_frames"] for e in eps),
        loophole=dict(loop),
        bytes_dataset=du(ds),
        bytes_total=du(out),
        **({} if aug_n is None else {"aug_episodes": aug_n}),
    )


def du(p: Path) -> int:
    n = 0
    for dp, _dn, fn in os.walk(p):
        for f in fn:
            with contextlib.suppress(OSError):
                n += os.lstat(os.path.join(dp, f)).st_size
    return n


def reason_of(log: Path, rc: int) -> str:
    """One-line failure reason: the log's last ✗/SystemExit/error line (rc if none). When that line points to a session
    or child log (…/*.log), that log's root line (root_line) is appended after ' ⇐ ': the bundle log only says 'died
    during the session', the cause is in the session log. The patterns read English log text only."""
    lines = log.read_text(errors="replace").splitlines() if log.is_file() else []
    pat = re.compile(
        r"(✗|Error|SystemExit|failed|died|dies|Traceback|"
        r"\[COLLECT\] .*(missing|is not|are not|stale|no ))"
    )
    hit = [ln for ln in lines if pat.search(ln)]
    tail = (hit or lines or [""])[-1].strip()
    sub = next((Path(m) for m in re.findall(r"(/\S+\.log)\b", tail) if Path(m).is_file()), None)
    root = root_line(sub) if sub else None
    return (
        f"rc {rc}"
        + (" (timeout)" if rc in (124, 137) else "")
        + f" · {tail[:240]}"
        + (f" ⇐ {root[:240]}" if root else "")
    )


def root_line(log: Path) -> str | None:
    """Root line of a child log: the last exception line (…Error:, SystemExit:; the traceback end), else the first ✗
    line · None when neither."""
    lines = [ln.strip() for ln in log.read_text(errors="replace").splitlines()]
    exc = [ln for ln in lines if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|SystemExit)\b.*:", ln)]
    if exc:
        return exc[-1]
    return next((ln for ln in lines if "✗" in ln), None)


def reason_key(reason: str) -> str:
    return re.sub(r"/\S+|\d+", "#", reason)[:120]
