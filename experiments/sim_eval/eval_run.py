"""Tab. 4 closed-loop evaluation of one fine-tuned checkpoint: run the L0/L1 scene lists of one task with the policy in
the loop, judge every rollout with the task judge and count successes per level →
`<out>/<run>_<step>[_<tag>]/<LEVEL>.json`.

Recipe (fixed, the paper's evaluation): evaluation profile `official` (gravity-compensated arm, frame-0 spawn of the
objects and extra slots), the policy acts until the end of the scene, the GR00T instruction text comes from the pinned
text map, P&P scenes show the red place pad, DP executes 8 actions per call and GR00T 16, and the task judge is the v2
judge (Pour, Sweep) or the P&P judge on top of it (Lift = P&P). A scene whose session fails is run once more (tag
`…b_r1`, spread over all cards); the retry wins. Technical failures are counted ('session failure'), never dropped.

Inputs are checked before any GPU work: the list sha256 (`lists/lists.sha256`), the text map sha256
(`lists/text_pin.json`) and its coverage of the listed scenes, L0 ⊆ training set and L1 ∩ training set = ∅
(`train_keys`), the checkpoint's own training set (DP `train_config.json`) and every job's episode inputs.

Per-row switches (each one proven by a log marker of every batch, `eval_checks`): `--texture-overlay` (training looks
without object textures), SPIDER runs start from SPIDER's own start (`--spider-start-dir`, required for `*_spider_*`
runs), DP server options `--dp-const-state-fix/--dp-const-state-eps` and `--dp-sampler/--dp-infer-steps`.
`--dry-run` prints the plan (tags, GPUs, ports, policy JSON, eval cfg) without starting anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from pathlib import Path

from experiments.sim_eval import batch as B
from experiments.sim_eval import eval_checks as CK
from experiments.sim_eval import eval_server as SV
from experiments.sim_eval import levels as LV
from experiments.sim_eval.eval_inputs import check_levels, check_training_set, server_options
from experiments.sim_eval.judges import pnp_batch as PB
from experiments.sim_eval.judges import run_v2 as RV2

HERE = Path(__file__).resolve().parent
LISTS = HERE / "lists"
TASKS = ("lift", "pour", "sweep")
#: Run name → policy data source (the row of the table). Order matters: the first match wins.
WHO = (
    ("_ours_", "ours"),
    ("_kin_", "kin"),
    ("_vis_", "vis"),
    ("_visft_", "vis"),
    ("_visdrop_", "vis"),
    ("_midft_", "mid"),
    ("_midvisft_", "midvis"),
    ("_m3b_", "m3b"),
    ("_g1c_", "g1c"),
    ("_v3ft_", "v3ft"),
    ("_v4ft_", "v4ft"),
    ("_v3midft_", "v3mid"),
    ("_v4midft_", "v4mid"),
    ("_spider_", "spider"),
)
PLACE_MARKER = {"lift": "pad-red"}
SPIDER_START = {
    "lift": "pnp_spider_start_v1.json",
    "pour": "pour_spider_start_v1.json",
    "sweep": "sweep_spider_start_v1.json",
}
TEXTURE_SHELLS = 97  # object shells of the texture overlay built for the textureless training looks


def log_to(path: Path):
    """Event logger: `[DPFIN] HH:MM <msg>` appended to the events log and echoed."""

    def ev(msg: str) -> None:
        line = f"[DPFIN] {time.strftime('%H:%M')} {msg}"
        with open(path, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    return ev


def who_of(run: str) -> str:
    """Data source of a run name."""
    for k, v in WHO:
        if k in run:
            return v
    raise SystemExit(f"[DPFIN] FAIL run name '{run}' names no data source ({', '.join(k for k, _ in WHO)})")


def step_of(policy: str, ckpt: str) -> str:
    """Six-digit training step of a checkpoint path (DP `…/checkpoints/<step>/pretrained_model`, GR00T
    `…/checkpoint-<n>`).
    """
    if policy == "gr00t":
        m = re.fullmatch(r"checkpoint-(\d+)", Path(ckpt).name)
        if not m:
            raise SystemExit(f"[DPFIN] FAIL GR00T checkpoint folder '{Path(ckpt).name}' is not checkpoint-<step>")
        return f"{int(m.group(1)):06d}"
    s = Path(ckpt).parent.name
    if not re.fullmatch(r"\d{6}", s):
        raise SystemExit(f"[DPFIN] FAIL checkpoint step folder '{s}' (…/checkpoints/<6 digits>/pretrained_model)")
    return s


def model_sha(policy: str, ckpt: str) -> str:
    """First 12 hex of the weights' sha256 (GR00T: of the concatenated shards)."""
    h = hashlib.sha256()
    files = sorted(Path(ckpt).glob("model-*.safetensors")) if policy == "gr00t" else [Path(ckpt) / "model.safetensors"]
    if not files or not all(f.is_file() for f in files):
        raise SystemExit(f"[DPFIN] FAIL checkpoint weights missing in {ckpt}")
    for f in files:
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
    return h.hexdigest()[:12]


def check_inputs(a, ev) -> dict:
    """Every input check that needs no GPU → the resolved plan inputs."""
    if a.task not in TASKS:
        raise SystemExit(f"[DPFIN] FAIL task '{a.task}'")
    who = who_of(a.run)
    step = step_of(a.policy, a.ckpt)
    dsr = check_training_set(a)
    levels = check_levels(a, ev, dsr, step)
    tagx = [a.tag] if a.tag else []
    cfg = {
        "profile": "official",
        "policy_until_t": True,
        "policy_job_text": True,
        "place_marker": PLACE_MARKER.get(a.task),
        "start_override": None,
        "render_overlay_root": None,
    }
    server_args = []
    if a.texture_overlay:
        m = json.loads((Path(a.texture_overlay) / "MANIFEST.json").read_text())
        if m.get("n_shell") != TEXTURE_SHELLS or m.get("root") != a.texture_overlay:
            raise SystemExit(f"[DPFIN] FAIL texture overlay {a.texture_overlay}: MANIFEST n_shell/root mismatch")
        cfg["render_overlay_root"] = a.texture_overlay
        tagx.append("texx")
    if who == "spider":
        if not a.spider_start_dir:
            raise SystemExit(
                "[DPFIN] FAIL a SPIDER run needs --spider-start-dir (SPIDER policies start from SPIDER's start)"
            )
        st = Path(a.spider_start_dir) / SPIDER_START[a.task]
        if not st.is_file():
            raise SystemExit(f"[DPFIN] FAIL SPIDER start file {st} not found")
        if a.texture_overlay:
            raise SystemExit("[DPFIN] FAIL SPIDER runs take no texture overlay")
        cfg["start_override"] = str(st)
    server_args, sampler = server_options(a, server_args, tagx)
    if a.policy == "dp" and not a.dp_python:
        raise SystemExit("[DPFIN] FAIL a DP run needs --dp-python (the lerobot environment)")
    marker = CK.place_marker_pattern(cfg["place_marker"]) if cfg["place_marker"] else ""
    return {
        "who": who,
        "step": step,
        "dsr": str(dsr),
        "levels": levels,
        "cfg": cfg,
        "server_args": server_args,
        "sampler": sampler,
        "tagx": "_".join(tagx),
        "marker": marker,
    }


def run_tag(a, P, tag: str, gpu: int, port: int, jobs: list[Path], W: Path, ev) -> set[str]:
    """One batch on one GPU: server up → sessions → server down → v2 judge of the rc-0 scenes → those scenes."""
    runs = Path(a.runs_root)
    cl = W / "cl" / tag
    pj = W / "policies" / f"{tag}.json"
    try:
        srv = (
            SV.start_dp(a.ckpt, P["dsr"], port, gpu, cl, a.dp_python, P["server_args"])
            if a.policy == "dp"
            else SV.start_gr00t(a.ckpt, P["dsr"], port, gpu, cl, f"ovf_{tag}", a.gr00t_root)
        )
    except RuntimeError as e:  # a failed server is a failed batch: its scenes go to the retry (or count as failures)
        ev(f"FAIL server {tag}: {e}")
        return set()
    SV.write_policy_json(pj, tag, srv, P["dsr"], a.text_map, a.policy_pylib)
    cfgp = W / "eval_cfg.json"
    spec = B.BatchSpec(
        name=tag,
        jobs=jobs,
        modes=[f"policy-{tag}", f"render-{tag}.policy-{tag}"],
        gpus=[gpu],
        per_gpu=a.per,
        runs_root=runs,
        min_avail_gb=a.min_avail_gb,
        run_scene_args=[
            "--policy",
            str(pj),
            "--eval-cfg",
            str(cfgp),
            "--timeout-s",
            str(a.session_timeout_s),
            "--isaaclab",
            a.isaaclab,
            "--augvis-assets",
            a.augvis_assets,
            *(["--arrays-pylib", a.arrays_pylib] if a.arrays_pylib else []),
        ],
    )
    try:
        res = B.run_batch(spec, stop=lambda: not srv.alive(), log=lambda m: print(m, flush=True))
    finally:
        ev(f"server stop {tag}: {srv.stop()}")
    if a.policy == "dp":
        if P["server_args"] and "--const-state-fix" in P["server_args"]:
            CK.const_state(srv.out / "server.log", tag)
        if P["sampler"]:
            CK.sampler(srv.out / "server.log", tag, P["sampler"])
    if P["cfg"]["render_overlay_root"]:
        ev(CK.texture_overlay(runs, tag, P["cfg"]["render_overlay_root"]))
    if P["cfg"]["start_override"]:
        ev(CK.spider_start(runs, tag, P["cfg"]["start_override"]))
    ok = B.rc0_scenes(Path(res["ledger"]))
    ok_jobs = [str(j) for j in jobs if B.scene_of(j) in ok]
    if ok_jobs:
        judged = RV2.judge_batch(
            tag,
            f"policy-{tag}",
            ok_jobs,
            runs,
            pj,
            a.data_root,
            a.home_map,
            W / "judge_tmp",
            log=_appender(cl / "judge.log"),
        )
        with open(W / "jfail.tsv", "a") as f:
            for sc, r in judged.items():
                if isinstance(r, str):
                    f.write(f"{sc}\t{r}\n")
    return ok


def _appender(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def w(msg: str) -> None:
        with open(path, "a") as f:
            f.write(msg + "\n")

    return w


def _write_level_runs(first, retry, a, W, lvl):
    """rc-0 scenes and judged run folders of every batch of the level → <W>/<lvl>.{rc0.txt,runs.txt,jfail.tsv}."""
    rc0, run_dirs = set(), []
    for _, tag, _ in first + retry:
        rc0 |= B.rc0_scenes(Path(a.runs_root) / f"{tag}.ledger.tsv")
        run_dirs.extend(
            d
            for d in sorted(Path(a.runs_root).glob(f"{tag}_*"))
            if d.is_dir() and (d / f"policy-{tag}" / "judge_v2.json").is_file()
        )
    (W / f"{lvl}.rc0.txt").write_text("".join(f"{s}\n" for s in sorted(rc0)))
    (W / f"{lvl}.runs.txt").write_text("".join(f"{d}\n" for d in run_dirs))
    (W / f"{lvl}.jfail.tsv").write_text((W / "jfail.tsv").read_text())
    return run_dirs


def _judge_pnp_level(a, pnp, run_dirs, P, ev, lv):
    """Lift: the P&P judge line of every judged run and the place-marker proof."""
    if a.task == "lift":
        ctx = PB.J._load()
        pnp.write_text(
            "".join(
                json.dumps(PB.judge_line(str(d), a.data_root, a.home_map, ctx), ensure_ascii=False) + "\n"
                for d in run_dirs
            )
        )
        nom = CK.place_marker(run_dirs, P["marker"])
        if nom:
            ev(f"FAIL {a.run} {P['step']} {lv} scenes without the place marker {len(nom)} — the level is invalid")


def run_level(a, P, lv: str, lv_i: int, W: Path, ev) -> None:
    """All batches of one level, the single retry, the P&P judge and the count → `<W>/<LEVEL>.json`."""
    lst, jobs = P["levels"][lv]
    lvl = lv.lower()
    gpus = [int(g) for g in a.gpus.split(",")]
    sfx = f"_{P['tagx']}" if P["tagx"] else ""
    base = f"dFE_dfin_{a.task}_{P['who']}{P['step']}_{lvl}"
    (W / "jfail.tsv").write_text("")
    first = [
        (g, f"{base}_g{g}{sfx}_r1", [j for i, j in enumerate(jobs) if i % len(gpus) == k]) for k, g in enumerate(gpus)
    ]
    first = [t for t in first if t[2]]
    for _, tag, _ in first:
        if (Path(a.runs_root) / f"{tag}.ledger.tsv").exists():
            raise SystemExit(f"[DPFIN] FAIL ledger {tag} exists — give --tag to run the same checkpoint again")
    ev(f"launch {a.run} {P['step']} {lv} · {len(jobs)} scenes · batches {len(first)}")
    ok = _parallel(
        [(lambda g=g, t=t, js=js: run_tag(a, P, t, g, 27000 + g * 100 + lv_i * 20, js, W, ev)) for g, t, js in first]
    )
    done = set().union(*ok) if ok else set()
    left = [j for j in jobs if B.scene_of(j) not in done]
    retry = []
    if left:
        for k, (g, tag, _) in enumerate(first):
            js = [j for i, j in enumerate(left) if i % len(first) == k]
            if js:
                retry.append((g, tag[: -len("_r1")] + "b_r1", js))
                ev(f"retry {retry[-1][1]} · scenes {len(js)} · GPU {g}")
        ok2 = _parallel(
            [
                (lambda g=g, t=t, js=js: run_tag(a, P, t, g, 27000 + g * 100 + lv_i * 20 + 10, js, W, ev))
                for g, t, js in retry
            ]
        )
        done2 = set().union(*ok2) if ok2 else set()
    bad = sum(len(js) for _, _, js in retry) - (len(done2) if left else 0)
    run_dirs = _write_level_runs(first, retry, a, W, lvl)
    pnp = W / f"{lvl}.pnp.jsonl"
    _judge_pnp_level(a, pnp, run_dirs, P, ev, lv)
    noatt, att = CK.extra_attach(run_dirs)
    if noatt:
        ev(f"FAIL {a.run} {P['step']} {lv} scenes without [EXTRA_ATTACH] {len(noatt)} — the level is invalid")
    ev(f"extra-attach {a.run} {P['step']} {lv} · present {len(run_dirs) - len(noatt)}/{len(run_dirs)} · attached {att}")
    res = LV.count_level(
        a.task,
        str(lst),
        str(W / f"{lvl}.runs.txt"),
        str(pnp),
        a.run,
        P["step"],
        lv,
        P["sha"],
        bad,
        str(W / f"{lvl}.rc0.txt"),
        str(W / f"{lvl}.jfail.tsv"),
        str(lst),
    )
    B.write_json(W / f"{lv}.json", res)
    if bad:
        ev(f"FAIL {a.run} {P['step']} {lv} session failures {bad} after the retry")
    ev(f"DONE {a.run} {P['step']} {lv} success {res['success']}/{res['n']} · {res['counts']} · {W / f'{lv}.json'}")


def _parallel(fns: list) -> list:
    out, errs = [None] * len(fns), []

    def go(i, f):
        try:
            out[i] = f()
        except BaseException as e:  # noqa: BLE001 — collected and re-raised after every card finished
            errs.append(e)
            out[i] = set()

    ts = [threading.Thread(target=go, args=(i, f)) for i, f in enumerate(fns)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0]
    return out


def parse(argv=None):
    """Command line of one evaluation run."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="run name, e.g. dense_dp_pour_ours_v2 (names the data source)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, help="training-set name (must match the checkpoint)")
    ap.add_argument(
        "--dataset-meta",
        required=True,
        help="training-set root holding meta/ (its parent holds the set for the key check)",
    )
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--gpus", required=True, help="comma list; the scene list is split over them")
    ap.add_argument("--per", type=int, default=1, choices=(1, 2, 3), help="sessions per GPU")
    ap.add_argument("--levels", default="L0,L1")
    ap.add_argument("--policy", default="dp", choices=("dp", "gr00t"))
    ap.add_argument("--lists", default=str(LISTS))
    ap.add_argument(
        "--scenes-home",
        required=True,
        help="scene stage folder: list entries <stage>/scenes/<set>/clips/<scene>/job.json resolve under it",
    )
    ap.add_argument("--text-map", required=True, help="instruction text map (sha256 pinned in <lists>/text_pin.json)")
    ap.add_argument("--runs-root", required=True, help="session outputs")
    ap.add_argument("--out", required=True, help="results: <out>/<run>_<step>[_<tag>]/<LEVEL>.json")
    ap.add_argument("--data-root", required=True, help="data root the judges read (scene trees)")
    ap.add_argument("--home-map", required=True, help="<HOME the scenes were built under>=<this HOME>")
    ap.add_argument("--isaaclab", required=True)
    ap.add_argument("--augvis-assets", required=True, help="random-look asset root of the render")
    ap.add_argument("--dp-python", default="", help="python of the DP (lerobot) environment")
    ap.add_argument("--policy-pylib", nargs="*", default=[], help="extra import dirs for the session's policy client")
    ap.add_argument(
        "--arrays-pylib", default="", help="import dir with pyarrow for the Isaac python (data-episode arrays)"
    )
    ap.add_argument("--gr00t-root", default="", help="GR00T server env root (experiments.policies.gr00t.server setup)")
    ap.add_argument("--tag", default="", help="suffix to run the same checkpoint again")
    ap.add_argument("--texture-overlay", default="", help="texture overlay root (rows trained on textureless looks)")
    ap.add_argument("--spider-start-dir", default="", help="folder of <task>_spider_start_v1.json (SPIDER runs)")
    ap.add_argument("--dp-const-state-fix", action="store_true")
    ap.add_argument("--dp-const-state-eps", type=float, default=0.0)
    ap.add_argument("--dp-sampler", default="", choices=("", "DDIM", "DDPM"))
    ap.add_argument("--dp-infer-steps", type=int, default=0)
    ap.add_argument("--session-timeout-s", type=int, default=3600)
    ap.add_argument("--min-avail-gb", type=int, default=None, help="host RAM floor for new sessions")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = log_to(out / "events.log")
    P = check_inputs(a, ev)
    W = out / f"{a.run}_{P['step']}{'_' + P['tagx'] if P['tagx'] else ''}"
    if a.dry_run:
        print(
            json.dumps(
                {
                    "out": str(W),
                    "who": P["who"],
                    "step": P["step"],
                    "cfg": P["cfg"],
                    "server_args": P["server_args"],
                    "levels": {k: [str(v[0]), len(v[1])] for k, v in P["levels"].items()},
                    "marker": P["marker"],
                },
                indent=1,
            )
        )
        return
    W.mkdir(parents=True, exist_ok=True)
    B.write_json(W / "eval_cfg.json", P["cfg"])
    P["sha"] = model_sha(a.policy, a.ckpt)
    ev(
        f"START {a.run} {P['step']} · {a.task} {P['who']} · policy {a.policy} · GPU {a.gpus} × {a.per} · "
        f"levels {a.levels} · "
        f"model sha {P['sha']} · set {a.dataset} · cfg {json.dumps(P['cfg'])} · server "
        f"{' '.join(P['server_args']) or '-'}"
    )
    for i, lv in enumerate(a.levels.split(",")):
        run_level(a, P, lv, i, W, ev)
    ev(f"end {a.run} {P['step']}")


if __name__ == "__main__":
    main()
