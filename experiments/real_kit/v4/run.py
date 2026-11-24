"""v4 canonical steps (start pose and approach) for one clip and a variant list: the functions of the v2 dense physics
chain called with the v4 switch on. The rule values live in rules.py.

Steps (subcommands):
  gen    one clip, variant list k → augment place_gen (--only, AUG_V4=1, AUG_DENSE_CELLS=1): variant trees, bake, pre
         gates. The judge cells (--cells) must be the cells that made the v2 data (read only; copies differ between
         machines), so at the end an independent oracle compares the v4 bake's arm cache with the v2 bake (--v2-shadow)
         byte for byte and dies on a difference.
  plan   variants that passed the bake → plan.py (approach schedule, check values)
  roll   the v2 rollout step (augment rollout, same checkpoint) + AUG_V4_APPROACH (approach physics, session hook) →
         post gate, retries (rules.RETRY_*) → roll_<mode>/retry.json
  v2ref  v2 reference of the join gate — the same variants once with the v2 RSI (warm-up mode; the approach only in
         warm-up batch 0) → v2ref/r0_s<RETRY_SEED0>
  stitch per round folder, stitch.py (approach rows + policy rows → stitched/run/rollouts.npz, v4_metrics.json — the
         render input; join gate = v2ref comparison, missing dies)
  all    gen → plan → roll → v2ref → stitch (stops when gen, plan or v2ref is non-zero; roll 6 = some variants
  failed
         within the cap — continues)
Writes only <root> and the shadow roots under it. Marker [AUG_V4_RUN] with tree path, commit and the all-object
collision value at every step (--all-obj-collision 0|1, required; v4 sets 1: the RL switch all_obj_collision, passed
to the rollout's launcher as `--set all_obj_collision=<v>`).
  python -m experiments.real_kit.v4.run <gen|plan|roll|v2ref|stitch|all> --root R --main <data root> --campaign C
      --bundle B --clip <base name> --ks 0,18,… --cells <judge cell root> --v2-shadow <v2 bake shadow> --gpu G
      --all-obj-collision 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[3]
MAIN = Path()  # canonical data root (--main; set in main())
PLANCOPY_TAG = "[AUG_V4_PLANCOPY]"  # marker: the clip plan was used without a per-round copy (round_plan)

os.chdir(T)
from dexengine.core import paths as P
from dexengine.record import bundle_cell as BC
from dexengine.record.augment import stage as AC
from experiments.real_kit.v4 import rules as V4
from experiments.real_kit.v4.run_plan import replan, say, step_plan

os.environ.setdefault(
    "DEXENGINE_RECORD_PLUGINS", "experiments.real_kit.v4.plugin"
)  # record hooks (dexengine.record.hooks)

assert AC.ROOT == T, AC.ROOT


def git(*c) -> str:
    return subprocess.run(["git", "-C", str(T), *c], capture_output=True, text=True).stdout.strip()


def v2_bake_sha(x, a, m: dict) -> int:
    """Independent oracle: are the bake files gen wrote (bake.bake_pair — tree bake output, two-hand IK, wrist cache)
    sha256-equal to the v2 pair, for every chosen variant (ks, ok | spare)? Different or missing → rc 7 (no
    re-bake)."""
    from experiments.real_kit.v4 import bake as V4B

    info = x.info
    lay = V4B.layout(x.hand, info["ds"], info["emb"], a.clip, V4.COMBO[0])
    same, bad = V4B.sha_oracle(lay, x.shadow, a.v2_shadow, MAIN, m, x.ks)
    say(
        f"{V4B.TAG} oracle {a.clip} · sha equal to the v2 pair {same} · different/missing {len(bad)} · v2 "
        f"{a.v2_shadow} (k 0 = canonical)"
    )
    for b in bad:
        say(f"{V4B.TAG} ✗ {b}")
    return 7 if bad or not same else 0


def setup(a, allobj: str) -> types.SimpleNamespace:
    """Common to all steps — tree and commit marker, judge cell, clip info, judge record md5, child env."""
    say(
        f"step {a.step} · tree {T} · commit {git('rev-parse', 'HEAD')[:12]} · "
        f"branch {git('rev-parse', '--abbrev-ref', 'HEAD')}"
        f" · dirty {bool(git('status', '--porcelain', '--', 'scripts', 'source'))} · AUG_V4_BASE="
        f"{os.environ.get('AUG_V4_BASE', '') or '(default)'} · ALL_OBJ={allobj}"
    )
    shadow = (a.root / "shadow").resolve()
    hand = V4.COMBO[1]
    cargs = types.SimpleNamespace(hand=hand, main=MAIN, gpu=str(a.gpu), tree=str(T))
    cell = BC.cell_of(cargs, a.campaign, a.bundle)
    clip = a.clip + P.TREE_SFX
    info = AC.clip_info(cell, clip)
    ks = [int(k) for k in a.ks.split(",") if k.strip()]
    d = a.root / "clips" / clip
    env = dict(
        os.environ,
        DEXENGINE_ARM=V4.COMBO[0],
        DEXENGINE_MAIN_ROOT=str(MAIN),
        CUDA_VISIBLE_DEVICES=str(a.gpu),
        AUG_DENSE_CELLS="1",
    )
    bundle = a.cells / a.campaign / a.bundle
    jf = [f for f in sorted((bundle / "judged").glob("*.jsonl")) if f'"clip": "{a.clip}"'.encode() in f.read_bytes()]
    if not jf:
        raise SystemExit(f"[AUG_V4_RUN] {bundle}/judged has no judge row for {a.clip}")
    say(
        f"judge cell bundle {bundle} · judge records "
        + " · ".join(f"{f.name} md5 {hashlib.md5(f.read_bytes()).hexdigest()}" for f in jf)
    )
    return types.SimpleNamespace(
        shadow=shadow,
        hand=hand,
        cargs=cargs,
        cell=cell,
        info=info,
        ks=ks,
        d=d,
        env=env,
        sets=("--set", f"all_obj_collision={allobj}"),
        bundle=bundle,
    )


def step_gen(a, x) -> int:
    d, info = x.d, x.info
    (d / "gen").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    AC.run_logged(
        [
            sys.executable,
            "dexengine/record/augment/place_gen.py",
            "--main",
            MAIN,
            "--shadow",
            x.shadow,
            "--hand",
            x.hand,
            "--dataset",
            info["ds"],
            "--emb",
            info["emb"],
            "--clip",
            a.clip,
            "--judged",
            x.bundle,
            "--only",
            ",".join(map(str, x.ks)),
            "--jobs",
            str(min(8, len(x.ks))),
            "--out",
            d / "gen",
            "--bake-from",
            a.v2_shadow,
            *(["--aerial"] if info["aerial"] else []),
        ],
        d / "gen.log",
        5400,
        x.env,
        "[AUG_GEN] DONE",
    )
    m = json.loads((d / "gen" / f"variants_{a.clip}.json").read_text())
    st = {}
    for v in m["variants"]:
        st[v["status"]] = st.get(v["status"], 0) + 1
    say(f"gen DONE {a.clip} · {time.time() - t0:.0f} s · state {st}")
    return v2_bake_sha(x, a, m)


def step_stitch(a, x) -> int:
    """Per round folder (those with a post gate) — approach rows + policy rows, variant check values."""
    d, info = x.d, x.info
    rds = sorted(p for p in (d / f"roll_{a.mode}{a.tag}").glob("r*_s*") if (p / "post.json").is_file())
    if not rds:
        raise SystemExit(f"[AUG_V4_RUN] {d}/roll_{a.mode}{a.tag} has no round that finished the post gate")
    ref = v2ref_dir(d)
    if not (ref / "run" / "rollouts.npz").is_file():
        raise SystemExit(f"[AUG_V4_RUN] join-stop gate reference {ref} missing — run the v2ref step first")
    for rdr in rds:
        pj = round_plan(d, rdr)
        AC.run_logged(
            [
                sys.executable,
                "experiments/real_kit/v4/stitch.py",
                "--rd",
                rdr,
                "--plan",
                pj,
                "--shadow",
                x.shadow,
                "--dataset",
                info["ds"],
                "--emb",
                info["emb"],
                "--v2-ref",
                ref,
            ],
            rdr / "stitch.log",
            900,
            x.env,
            "[AUG_V4_STITCH] DONE",
        )
    say(f"stitch DONE {a.clip} · rounds {len(rds)} · {[p.name for p in rds]}")
    return 0


def round_plan(d: Path, rdr: Path) -> Path:
    """The schedule a round used = its copy rdr/plan.json (left by one_round). A round without a copy uses the clip plan
    with a marker, and dies if that clip has re-planned variants (*.f0.npz), since the clip plan is then not that
    round's."""
    if (rdr / "plan.json").is_file():
        return rdr / "plan.json"
    f0 = sorted(p.name for p in (d / "plan").glob("*.f0.npz"))
    if f0:
        raise SystemExit(
            f"[AUG_V4_RUN] {rdr}: no round copy of plan.json · re-planned variants {f0} — the clip plan is "
            "not the schedule that round used"
        )
    say(f"{PLANCOPY_TAG} {rdr.name} no round copy → clip plan (round from before per-round plan copies)")
    return d / "plan" / "plan.json"


def v2ref_dir(d: Path) -> Path:
    return d / "v2ref" / f"r0_s{V4.RETRY_SEED0}"


def step_v2ref(a, x, m: dict, okn: list) -> int:
    """v2ref — the v2 reference of the join gate (stitch.join_gate): variants with a schedule, one round in warm-up
    mode
    (seed RETRY_SEED0). Only warm-up batch 0 approaches and the judge drops it — the real batches are the v2 RSI."""
    d = x.d
    pl = json.loads((d / "plan" / "plan.json").read_text())
    names = [n for n in okn if pl.get(n, {}).get("K")]
    rdr = v2ref_dir(d)
    if rdr.exists():
        raise SystemExit(f"[AUG_V4_RUN] {rdr} already exists — the reference is rolled once (delete it and rerun)")
    os.environ.update(AUG_V4_APPROACH=str(d / "plan" / "plan.json"), AUG_V4_APPROACH_MODE="warmup")
    t1 = time.time()
    res = one_round(a, x, m, rdr, names, V4.RETRY_SEED0)
    say(
        f"v2ref DONE {a.clip} · variants {len(names)} · episodes {sum(r[0] for r in res.values())} · "
        f"{time.time() - t1:.0f}s · {rdr}"
    )
    return 0


def one_round(a, x, m: dict, rdr: Path, names: list[str], seed: int) -> dict:
    """One round — roll out `names` only (seed) + post gate → {variant: (attempts, successes, post passes)}."""
    (rdr / "gen").mkdir(parents=True, exist_ok=True)
    vs = [
        dict(v, status="ok") if v["name"] in names else (v if v["status"] == "reject" else dict(v, status="phys_other"))
        for v in m["variants"]
    ]
    vji = rdr / "gen" / f"variants_{a.clip}.json"
    vji.write_text(json.dumps(dict(m, variants=vs), indent=1))
    shutil.copyfile(
        x.d / "plan" / "plan.json", rdr / "plan.json"
    )  # the schedule this round used (also right after a re-plan)
    os.environ.update(AUG_V4_APPROACH_OUT=str(rdr / "approach"), AUG_V4_SEED=str(int(seed)))
    (rdr / "approach").mkdir(parents=True, exist_ok=True)
    V, spec, run = AC.rollout(x.cargs, x.cell, vji, rdr, x.shadow, launch_args=x.sets)
    pj = rdr / "post.json"
    if not pj.is_file():
        noref = [] if any(int(v["k"]) == 0 for v in m["variants"] if v["name"] in names) else ["--no-ref"]
        if (
            noref
        ):  # the original placement (pv00) is not in this round — judge thresholds only (the stricter side) · marker
            say(f"{a.clip}: round {rdr.name} has no pv00 → post gate --no-ref (judge thresholds only)")
        AC.run_logged(
            [
                sys.executable,
                "dexengine/record/augment/coll_gate.py",
                "post",
                "--variants",
                vji,
                "--main",
                MAIN,
                "--shadow",
                x.shadow,
                "--run",
                run,
                "--out",
                pj,
                *noref,
            ],
            rdr / "post.log",
            AC.POST_TIMEOUT_S,
            x.env,
            "[AUG_COLL] post",
        )
    post = json.loads(pj.read_text())
    pok = {P.tree_name(r["name"]) for r in post["kept"] if r["post_ok"]}
    z = np.load(rdr / "run" / "rollouts.npz", allow_pickle=False)
    cn = [P.tree_name(str(x_)) for x_ in z["clip_names"]]
    et = z["ep_table"]
    real = et[et[:, 3] == 0]
    out = {}
    for n in names:
        tn = P.tree_name(n)
        ci = cn.index(tn)
        mm = real[real[:, 0] == ci]
        out[n] = (int(len(mm)), int(mm[:, 5].sum()), tn in pok, V4.ok_rank(post["kept"], tn))
    return out


def step_roll(a, x, m: dict, okn: list) -> int:
    """roll — the v2 rollout step (augment rollout) as is + the approach physics hook (session env), retries →
    retry.json.
    """
    d = x.d
    if a.mode not in ("warmup", "every", "every_pin", "every_rsi"):
        raise SystemExit(f"[AUG_V4_RUN] --mode {a.mode!r} — warmup|every|every_pin|every_rsi only")
    pl = json.loads((d / "plan" / "plan.json").read_text())
    pfail = {n: pl.get(n, {}).get("fail", "no schedule") for n in okn if pl.get(n, {}).get("fail") or n not in pl}
    if (
        pfail
    ):  # variants without a schedule are left out (the session dies on a clip missing from the schedule) · marker
        say(f"{a.clip}: schedule failures {len(pfail)} left out — {pfail}")
        okn = [n for n in okn if n not in pfail]
    rd = d / f"roll_{a.mode}{a.tag}"
    rd.mkdir(parents=True, exist_ok=True)
    os.environ.update(AUG_V4_APPROACH=str(d / "plan" / "plan.json"), AUG_V4_APPROACH_MODE=a.mode)
    # retry rule (rules.RETRY_*): each variant rolls again with a new seed until one success passes the post gate;
    # attempt cap
    # RETRY_MAX_ATTEMPTS (episodes); a variant with 0 within the cap stays in the table as failed
    stat = {
        n: {"attempts": 0, "successes": 0, "post_ok": False, "rounds": [], "cand_attempts": 0, "replan": None}
        for n in okn
    }
    t1 = time.time()
    r = 0
    while True:
        need = [n for n in okn if not stat[n]["post_ok"] and stat[n]["cand_attempts"] < V4.RETRY_MAX_ATTEMPTS]
        if not need:
            need = replan(a, x, okn, stat)
        if not need:
            break
        seed = V4.RETRY_SEED0 + r
        res = one_round(a, x, m, rd / f"r{r}_s{seed}", need, seed)
        for n, (att, suc, ok, okr) in res.items():
            st = stat[n]
            st["attempts"] += att
            st["cand_attempts"] += att
            st["successes"] += suc
            st["rounds"].append(
                {"round": r, "seed": seed, "attempts": att, "successes": suc, "post_ok": ok, "ok_rank": okr}
            )
            if (
                ok and not st["post_ok"]
            ):  # variant trajectory = this round's first passing rank (rules.ok_rank, read by the render)
                st["post_ok"], st["ok_round"], st["ok_rank"] = True, r, okr
        print(
            f"{V4.RETRY_TAG} {a.clip} round {r} (seed {seed}) · variants {len(need)} · passed this round "
            f"{sum(1 for n in need if res[n][2])} · passed in total "
            f"{sum(s_['post_ok'] for s_ in stat.values())}/{len(okn)}",
            flush=True,
        )
        r += 1
    (rd / "retry.json").write_text(
        json.dumps(
            {
                "mode": a.mode,
                "max_attempts": V4.RETRY_MAX_ATTEMPTS,
                "replan_floor_off": V4.REPLAN_FLOOR_OFF,
                "seed0": V4.RETRY_SEED0,
                "plan_fail": pfail,
                "variants": stat,
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    nok = sum(s_["post_ok"] for s_ in stat.values())
    zero = [n for n, s_ in stat.items() if not s_["post_ok"]]
    say(
        f"roll DONE {a.clip} · branch {a.mode} · variants {len(okn)} · post pass {nok}/{len(okn)} · rounds {r} · "
        f"0 successes within the cap {zero} · schedule failures {list(pfail)} · {time.time() - t1:.0f}s · {rd}"
    )
    return 0 if not zero and not pfail else 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("gen", "plan", "roll", "v2ref", "stitch", "all"))
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument(
        "--cells", required=True, type=Path, help="judge cell root that made v2 (parent of <campaign>/<bundle>)"
    )
    ap.add_argument("--v2-shadow", required=True, type=Path, help="v2 bake shadow root (oracle check · read only)")
    ap.add_argument("--ks", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--mode", default="", help="roll: approach physics branch warmup|every|every_pin|every_rsi")
    ap.add_argument(
        "--tag", default="", help="roll · stitch: run folder suffix (default _on|_off = the all-object collision value)"
    )
    ap.add_argument("--main", type=Path, required=True, help="canonical data root the v2 cells and trees are read from")
    ap.add_argument(
        "--all-obj-collision",
        required=True,
        choices=("0", "1"),
        help="RL switch all_obj_collision of the rollouts (v4 set = 1)",
    )
    a = ap.parse_args()
    global MAIN
    MAIN = a.main.resolve()
    if os.environ.get("AUG_V4") != "1":
        raise SystemExit("[AUG_V4_RUN] AUG_V4 is not 1 — v4 steps run only with the v4 switch on")
    allobj = a.all_obj_collision
    if not a.tag:  # rollout folder suffix = the all-object collision value (roll_<mode>_<on|off>)
        a.tag = "_on" if allobj == "1" else "_off"
    if a.step == "all":
        for st in ("gen", "plan", "roll", "v2ref", "stitch"):
            rc = subprocess.run([sys.executable, __file__, st, *sys.argv[2:]]).returncode
            say(f"all · {st} rc={rc}")
            if rc and (st in ("gen", "plan", "v2ref") or (st == "roll" and rc != 6)):
                return rc
        return 0
    x = setup(a, allobj)
    if a.step == "gen":
        return step_gen(a, x)
    m = json.loads((x.d / "gen" / f"variants_{a.clip}.json").read_text())
    okn = [v["name"] for v in m["variants"] if v["status"] in ("ok", "spare") and int(v["k"]) in x.ks]
    if a.step == "plan":
        return step_plan(a, x, okn)
    if a.step == "v2ref":
        return step_v2ref(a, x, m, okn)
    if not a.mode:  # canonical hand-off (rules.HANDOFF_MODE) — other modes are explicit diagnostic choices
        a.mode = V4.HANDOFF_MODE
        say(f"branch = canonical handoff {a.mode} (experiments.real_kit.v4.rules.HANDOFF_MODE)")
    if a.step == "stitch":
        return step_stitch(a, x)
    return step_roll(a, x, m, okn)


if __name__ == "__main__":
    sys.exit(main())
