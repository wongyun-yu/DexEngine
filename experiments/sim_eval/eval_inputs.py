"""Input checks of the Tab. 4 evaluation that need no GPU (parts of eval_run.check_inputs): the training set and
checkpoint of --dataset, the pinned lists and text map per level with the train-set split and episode inputs, and the
policy-server options."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.sim_eval import train_keys as TK


def check_training_set(a):
    """The training-set meta and the checkpoint belong to --dataset; returns the meta folder."""
    dsr = Path(a.dataset_meta)
    if dsr.name != a.dataset:
        raise SystemExit(f"[DPFIN] FAIL --dataset-meta '{dsr}' is not the set {a.dataset}")
    if not ((dsr / "meta/info.json").is_file() and (dsr / "meta/stats.json").is_file()):
        raise SystemExit(f"[DPFIN] FAIL training-set meta missing in {dsr}/meta")
    if a.policy == "dp":
        tr = json.loads((Path(a.ckpt) / "train_config.json").read_text())["dataset"]["root"]
        if Path(tr).name != a.dataset:
            raise SystemExit(f"[DPFIN] FAIL checkpoint training data '{tr}' is not the set {a.dataset}")
    elif not (Path(a.ckpt) / "config.json").is_file():
        raise SystemExit(f"[DPFIN] FAIL GR00T checkpoint {a.ckpt} has no config.json")
    return dsr


def check_levels(a, ev, dsr, step):
    """Pinned text map, list sha256, train-set split and episode inputs per level; returns {level: (list, jobs)}."""
    pin = json.loads((Path(a.lists) / "text_pin.json").read_text())
    tsha = hashlib.sha256(Path(a.text_map).read_bytes()).hexdigest()
    if tsha != pin["sha256"]:
        raise SystemExit(f"[DPFIN] FAIL text map sha256 {tsha[:12]} ≠ pinned {pin['sha256'][:12]}")
    tmap = json.loads(Path(a.text_map).read_text())
    shas = dict(reversed(ln.split()) for ln in (Path(a.lists) / "lists.sha256").read_text().splitlines() if ln.strip())
    levels = {}
    for lv in a.levels.split(","):
        lst = Path(a.lists) / f"{a.task}_{lv.lower()}.txt"
        if hashlib.sha256(lst.read_bytes()).hexdigest() != shas.get(lst.name):
            raise SystemExit(f"[DPFIN] FAIL list sha256 ({lst})")
        ev(TK.check(a.dataset, lv, lst, dsr.parent).replace("[TRAIN_KEYS] OK", f"train-set check {a.run} {step} {lv}"))
        entries = [x.strip() for x in lst.read_text().splitlines() if x.strip()]
        miss = [TK.scene_name(x) for x in entries if TK.scene_name(x) not in tmap]
        if miss:
            raise SystemExit(f"[DPFIN] FAIL the text map does not cover {lst}: {len(miss)} {miss[:3]}")
        jobs = [
            Path(a.scenes_home).joinpath(*Path(x).parts[1:]) for x in entries
        ]  # entries start with the stage folder name
        for j in jobs:
            ep = json.loads(j.read_text())["episode"]
            if not (Path(ep["files"]["forces"]).is_file() and Path(ep["root_dir"], "meta/info.json").is_file()):
                raise SystemExit(f"[DPFIN] FAIL episode inputs (forces · info) of {j} not found")
        levels[lv] = (lst, jobs)
    return levels


def server_options(a, server_args, tagx):
    """Policy-server options (GR00T env, DP const-state fix, DP sampler); returns the sampler label."""
    if a.policy == "gr00t" and not a.gr00t_root:
        raise SystemExit("[DPFIN] --policy gr00t needs --gr00t-root (the GR00T server env)")
    if a.policy == "gr00t" and (a.dp_const_state_fix or a.dp_sampler):
        raise SystemExit("[DPFIN] FAIL DP server options given for a GR00T run")
    if a.dp_const_state_fix:
        server_args.append("--const-state-fix")
        if a.dp_const_state_eps:
            server_args += ["--const-state-eps", str(a.dp_const_state_eps)]
    elif a.dp_const_state_eps:
        raise SystemExit("[DPFIN] FAIL --dp-const-state-eps needs --dp-const-state-fix")
    sampler = ""
    if a.dp_sampler or a.dp_infer_steps:
        if not (a.dp_sampler and a.dp_infer_steps):
            raise SystemExit("[DPFIN] FAIL --dp-sampler and --dp-infer-steps go together")
        server_args += ["--sampler", a.dp_sampler, "--infer-steps", str(a.dp_infer_steps)]
        sampler = f"{a.dp_sampler} steps={a.dp_infer_steps}"
        tagx.append(f"{a.dp_sampler.lower()}{a.dp_infer_steps}")
    return server_args, sampler
