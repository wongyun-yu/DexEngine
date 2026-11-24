"""Run one closed-loop evaluation scene: one Isaac session (`session.py`) on one GPU → `<out>/<mode>/episode.npz`.

`python -m experiments.sim_eval.run_scene --job <job.json> --out <dir> --gpu <id> --modes <m1,m2> [--cap-mult 1.5]
 [--policy <policy.json>] [--eval-cfg <eval_cfg.json>] [--timeout-s 3600] --isaaclab <IsaacLab> --augvis-assets <dir>`

job_run.json (written to `<out>`) = the scene's job.json + out, modes, cap_mult, episode_arrays (the data episode's
arrays, built once per episode under `--arrays-dir`), policy or policies (from the policy JSON: one `{kind: …}` or
`{"policies": {name: {kind: …}}}` → modes `policy-<name>`) and `eval` (the session options, see session_cfg).

Evaluation profile (required): `official` = gravity-compensated arm without reference velocity lead
(RL switch arm_ff_vel_scale = 0), objects spawned and extra slots held at frame 0 (spawn_f0 = extra_f0 = 1). Without a
profile the env default controller would add the reference lead on top of the stored labels (a double lead). RL
switches (`dexengine.rl.tasks.switches`) go to job_run.json `switches`; the session writes them over the dump's
switches file (`switches.derive`). Option all_obj_collision → switch all_obj_collision; render_overlay_root →
DEXENGINE_RENDER_OVERLAY_ROOT and `--augvis-assets` → AUGVIS_ASSETS (environment of the Isaac process).

Exit code: 0 ok · 2 setup error · 3 no `[E5_EVAL] DONE` line · 5 marker counts (every physics mode must log
`[E5_ARMCTRL]`, `[E5_PHYS]` and `[E5_MARKERS]`) · else the session's own code. Last line `[E5_RUN] RC=<n> …`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARK = "[E5_RUN]"
PROFILES = {"official": {"switches": {"arm_ff_vel_scale": 0.0}, "eval": {"spawn_f0": "1", "extra_f0": "1"}}}
#: eval_cfg keys passed to the Isaac process as dexengine environment variables.
DEXENGINE_ENV = {"render_overlay_root": "DEXENGINE_RENDER_OVERLAY_ROOT"}
#: eval_cfg keys that set an RL switch (field of dexengine.rl.tasks.switches.RLSwitches).
SWITCH_KEYS = ("all_obj_collision",)


def eval_options(cfg: dict) -> tuple[dict, dict, dict]:
    """eval_cfg JSON → (session options as strings, extra environment of the Isaac process, RL switches)."""
    prof = cfg.get("profile")
    if prof not in PROFILES:
        raise SystemExit(f"{MARK} eval profile {prof!r} — one of {sorted(PROFILES)} (required)")
    opts, env, sw = dict(PROFILES[prof]["eval"]), {}, dict(PROFILES[prof]["switches"])
    for k, v in cfg.items():
        if v is None or v is False or v == "":
            continue
        s = "1" if v is True else str(v)
        if k in DEXENGINE_ENV:
            env[DEXENGINE_ENV[k]] = s
        elif k in SWITCH_KEYS:
            sw[k] = s
        else:
            opts[k] = s
    return opts, env, sw


def job_run(
    job: Path,
    out: Path,
    modes: list[str],
    cap: float,
    arrays: Path,
    policy: Path | None,
    opts: dict,
    sw: dict | None = None,
) -> dict:
    """job_run.json content."""
    j = json.loads(job.read_text())
    j.update(out=str(out), modes=modes, cap_mult=cap, episode_arrays=str(arrays), eval=opts, switches=dict(sw or {}))
    if policy is not None:
        pj = json.loads(policy.read_text())
        if "kind" in pj:
            j["policy"] = pj
        elif (
            isinstance(pj.get("policies"), dict)
            and pj["policies"]
            and all("kind" in v for v in pj["policies"].values())
        ):
            j["policies"] = pj["policies"]
        else:
            raise SystemExit(f"{MARK} policy JSON {policy} — one {{kind}} or policies{{name: {{kind}}}}")
    return j


def episode_arrays(isa_py: str, job: Path, arrays_dir: Path, pylib: str) -> Path:
    """The data episode's arrays (built once per episode key)."""
    from experiments.sim_eval.episode_arrays import episode_key

    arr = arrays_dir / f"{episode_key(json.loads(job.read_text())['episode'])}.npz"
    if not arr.is_file():
        arrays_dir.mkdir(parents=True, exist_ok=True)
        cmd = [isa_py, "-m", "experiments.sim_eval.episode_arrays", "--job", str(job), "--out-dir", str(arrays_dir)]
        path = os.pathsep.join([str(ROOT)] + ([pylib] if pylib else []))
        subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=path), check=False)
    if not arr.is_file():
        raise SystemExit(f"{MARK} data arrays {arr} were not built")
    return arr


def commit() -> str:
    """Code identity of this tree: git HEAD, else the E5_COMMIT file of a package tree (no .git), else 'unknown'
    (a run without a code identity is not accepted as a result)."""
    p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    if p.stdout.strip():
        return p.stdout.strip()
    f = ROOT / "E5_COMMIT"
    return (f.read_text().strip() if f.is_file() else "") or "unknown"


def marker_rc(log: str, modes: list[str]) -> int:
    """3 without the DONE line, 5 when a physics mode lacks its controller/physics/marker lines, else 0."""
    if "[E5_EVAL] DONE" not in log:
        return 3
    nphys = sum(1 for m in modes if m != "gate" and not m.startswith("render-"))
    lines = log.splitlines()
    nctrl, nphy, nmrk = (sum(1 for x in lines if k in x) for k in ("[E5_ARMCTRL]", "[E5_PHYS]", "[E5_MARKERS]"))
    return 5 if (nctrl != nphys or nphy < nphys or nmrk < nphys) else 0


def parse(argv=None):
    """Command line of one scene session."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--cap-mult", type=float, default=1.5)
    ap.add_argument("--policy", default="")
    ap.add_argument("--eval-cfg", default="", help="evaluation options (profile required)")
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--isaaclab", required=True)
    ap.add_argument("--augvis-assets", required=True, help="random-look asset root (manifest.json)")
    ap.add_argument("--arrays-dir", default="", help="data-episode array cache (default <out>/../_arrays)")
    ap.add_argument("--arrays-pylib", default="", help="import dir with pyarrow for the Isaac python (array builder)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    out, job = Path(a.out), Path(a.job)
    modes = [m for m in a.modes.split(",") if m]
    if not job.is_file():
        print(f"{MARK} job {job} not found", file=sys.stderr)
        return 2
    if not (Path(a.augvis_assets) / "manifest.json").is_file():
        print(f"{MARK} random-look assets {a.augvis_assets}/manifest.json not found", file=sys.stderr)
        return 2
    cfg = json.loads(Path(a.eval_cfg).read_text()) if a.eval_cfg else {}
    opts, xenv, sw = eval_options(cfg)
    out.mkdir(parents=True, exist_ok=True)
    isa_py = str(Path(a.isaaclab) / "env_isaaclab/bin/python")
    arr = episode_arrays(isa_py, job, Path(a.arrays_dir) if a.arrays_dir else out.parent / "_arrays", a.arrays_pylib)
    (out / "job_run.json").write_text(
        json.dumps(
            job_run(job, out, modes, a.cap_mult, arr, Path(a.policy) if a.policy else None, opts, sw),
            indent=1,
            ensure_ascii=False,
        )
    )
    tmp = Path(tempfile.mkdtemp(prefix=f"e5_{out.name}_"))
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_VERSION")}
    env.update(
        xenv,
        E5_JOB=str(out / "job_run.json"),
        CUDA_VISIBLE_DEVICES=str(a.gpu),
        PYTHONPATH=str(ROOT),
        ISAACLAB_DIR=a.isaaclab,
        AUGVIS_ASSETS=a.augvis_assets,
        TMPDIR=str(tmp),
        XDG_RUNTIME_DIR=str(tmp / "xdg"),
        WANDB_MODE="disabled",
        WANDB_DIR=str(tmp / "wandb"),
        ACCEPT_EULA="Y",
        PRIVACY_CONSENT="Y",
        OMNI_KIT_ACCEPT_EULA="YES",
        OMNI_KIT_ALLOW_ROOT="1",
        PYTHONUNBUFFERED="1",
        PYTHONFAULTHANDLER="1",
    )
    (tmp / "xdg").mkdir(mode=0o700)
    (tmp / "wandb").mkdir()
    print(
        f"{MARK} code {ROOT} @ {commit()} · GPU {a.gpu} · modes {a.modes} · cap ×{a.cap_mult} · job {job} "
        f"· out {out} · "
        f"eval {json.dumps(opts)}",
        flush=True,
    )
    rc = 0
    with open(out / "session.log", "w") as lf:
        try:
            p = subprocess.run(
                [
                    "timeout",
                    "-k",
                    "60",
                    str(a.timeout_s),
                    "bash",
                    "-c",
                    'source scripts/_isaac_env.sh && exec "$ISAACLAB_DIR/isaaclab.sh" -p '
                    "experiments/sim_eval/session.py",
                ],
                cwd=ROOT,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            rc = p.returncode
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    log = (out / "session.log").read_text(errors="replace")
    if rc == 0:
        rc = marker_rc(log, modes)
    print(f"{MARK} RC={rc} · {log.count('[E5_EVAL]')} markers · {out / 'session.log'}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
