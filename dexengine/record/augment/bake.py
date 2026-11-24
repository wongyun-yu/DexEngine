"""Placement augment: builds the arm trajectory of a variant **the same way as the bake** (called by
augment.place_gen). Variants k > 0 only: the canonical cell pv00 is not baked again, it is the canonical clip's own
bake outputs copied byte for byte (augment.place_src canon_variant) — a bake again from
the judge seed re-solves the arm on another machine (DLS polish · cross repair · retime) and can leave the arm path the
judge passed.

Why: putting the judge q on the canonical timetable made judge L's IK branch differ from the original judge (1/3 of
the sides), and even the canonical cell failed the speed gate. The bake refines the judge q as a seed and retimes with
its own q.
Path (the step list of the bake chain scripts/stage2/chain_fast.sh without its first `bake`, through the same resident
  worker dexengine/placement/bake/server.py (and its `send` client) -> chain.run_specs; no copy; order check =
  tests/record test_augment_bake_steps_follow_the_bake_chain_order, which reads chain_fast.sh):
    body -> wrist x2 -> ik x2 (stage_dp · --reuse-npz = seed) [-> idle side body · wrist] [-> xr] -> retime -> body
    -> wrist x2
  The variant tree replaces the `bake` step (placement put on the f0 staging): the **tree the IK saw** (base_kin) of
  that clip's canonical tree, rigidly moved by the variant placement (augment.place_src make_variant · canonical
  rigid_pl).
Input tree rule (base_kin; not a fallback): when the canonical tree has an xr report (cross_repair.REPORT_REL) the bake
  ran the cross repair -> its input copy `.pre_xr.npz` is the tree the IK saw (missing = dies) and the xr step runs
  too. Otherwise `.pre_rt.npz` (before retiming) and no xr.
Seeds (seeds: the same rule as the bake chain; only sides with a file; idle sides take stage_dp's rest pose path):
  judge mode L's q of that cell (bundle ikc/ · extra cells: their sub-bundle). k = 0 has no seed (not baked).
Value owners (not written again): safety factor = the canonical tree's retime_safety (that clip's bake value) ·
  stretch cap = the stage_dp parser default (the same as judge --dp-max-stretch) · IK pool = paths.pool_root · height =
  adopted placement robot_z_off · xr assets = parent of the assets_objects layer (as bake_drive) · env = canon_env.sh +
  the bake axes (active side mode = the source chunk judge.log).
Support stamp: steps that write tree trajectories call coords.stamp_scene right after (that function's rule); after
  writing the variant tree and before the bake steps it is stamped again (keep_height · the bake_post worker);
  otherwise the stage_dp and xr guards see the support disc of the canonical placement.
Later gates: length (here · dexengine.placement.bake.chain.halfwrite_lengths) · contact · gate speed · placement
(augment.bake_post) ·
  final_guard (augment.coll_gate); the same functions as the chain. pv00: the copy check (augment.bake_post
  canon_copy_check) in place of the bake.
Markers: [AUG_BAKE] … · server pool [AUG_BAKE] pool …
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from dexengine.core import paths as P
from dexengine.core.errors import input_missing
from dexengine.core.repo import REPO_ROOT
from dexengine.record.augment.scene import shadow_dir

MARK = "[AUG_BAKE]"
ROOT = REPO_ROOT  # tree root
#: resident server boot deadline (s; measured boot ≈ 10 to 20 s) · one variant deadline (s; measured incl. the
#: general DP fallback, with room).
BOOT_S = 300
#: resident server socket place: the same short root as the canonical bake (bake_drive.sh `mktemp -d
#: /tmp/bakepool.XXXXXX`). Following TMPDIR, deep job scratch places (nodes, pods) exceed the AF_UNIX path limit
#: (108 B) and the server dies in bind.
SOCK_BASE = "/tmp"
VARIANT_S = 900
#: env of the chain's final_guard ledger and the cross repair ledger: the augment does not write canonical ledgers
#: (removed from the server and client env).
NO_LEDGER_ENV = ("DEXENGINE_FINAL_GUARD_LEDGER", "DEXENGINE_XR_LEDGER")


def die(msg: str):
    raise SystemExit(f"{MARK} {msg}")


def bake_code_files() -> tuple:
    """Bake step code that decides variant outputs (repo relative): the step table owner
    dexengine.placement.bake.chain.SCRIPTS (+ the modules split off a step, chain.STAGE_PARTS) + the run frame (server,
    client, runner). Read by the augment cache fingerprint (augment.stage fingerprint; the list is not written
    again)."""
    from dexengine.placement.bake import chain as SR

    # run frame: chain (run_specs) · server (the worker and its `send` client)
    return (
        tuple(SR.SCRIPTS.values())
        + SR.STAGE_PARTS
        + tuple(f"dexengine/placement/bake/{n}.py" for n in ("chain", "server"))
    )


# ── inputs · values ──────────────────────────────────────────────
def xr_report(tree: Path) -> Path:
    """The tree's xr report place: read from the source of its value owner cross_repair.REPORT_REL (loading the module
    pulls in warp and the GPU)."""
    import ast

    src = ast.parse((ROOT / "dexengine/placement/repair/cross_repair.py").read_text())
    for n in src.body:
        if isinstance(n, ast.Assign) and [getattr(t, "id", "") for t in n.targets] == ["REPORT_REL"]:
            return Path(tree) / ast.literal_eval(n.value)
    return die("cross_repair.REPORT_REL not found")


def base_kin(src) -> tuple[dict, bool, str]:
    """The tree the canonical tree's IK saw -> (array, whether the cross repair ran, source file name). Rule in the
    module doc."""
    k0 = src.tree / "0/trajectory_kinematic.npz"
    xr = xr_report(src.tree).is_file()
    f = k0.with_suffix(".pre_xr.npz") if xr else k0.with_suffix(P.IK_SIDECAR_SFX)
    if not f.is_file():
        die(
            input_missing(
                f"{src.tree.name}: {'xr report present but ' if xr else ''}the tree the IK saw {f.name} is "
                "missing: not a bake output"
            )
        )
    z = dict(np.load(f, allow_pickle=True))
    T = len(src.kin["retime_tau"])
    if len(z["qpos"]) != T:
        die(f"{src.tree.name}: {f.name} rows {len(z['qpos'])} != T before retiming {T}")
    return z, xr, f.name


def values(src) -> dict:
    """Chain argument values, from their owners (module doc)."""
    from dexengine.placement.coarse_path import dp as stage_dp

    saf = float(np.asarray(src.kin["retime_safety"]))
    return {
        "safety": saf,
        "max_stretch": float(stage_dp.build_parser().get_default("max_stretch")),
        "pool": str(P.pool_root()),
        "zoff": float(src.adopted["robot_z_off"]),
        "idle": [str(s) for s in src.adopted.get("idle_sides") or []],
        "assets": str(P.layer_path("assets_objects", src.ds).parent),
    }


def clip_ctx(src) -> dict:
    """The judge chunk that baked the clip (adopted row source_tag; the same window as the cells run reading judge
    conditions) -> chunk tag · judge active side mode (what stage2 passed to the bake as DEXENGINE_ACTIVE_SIDES)."""
    from dexengine.record.augment import cells as R

    clip = src.tree_name.removesuffix(P.TREE_SFX)
    row = R.adopted_rows(src.hand, src.ds, [clip])[clip]
    ck = R.chunk_record(src.hand, row["source_tag"])
    return {"tag": row["source_tag"], "active": ck["active"]}


def seeds(src, k: int, bundle: Path) -> dict:
    """Side -> seed path (only sides that have one) of a variant k > 0; rule in the module doc."""
    clip = src.tree_name.removesuffix(P.TREE_SFX)
    if k == 0:
        die(f"{clip}: the canonical cell pv00 has no bake seed (a copy of the canonical clip, place_src canon_variant)")
    from dexengine.record.augment import judged as AJ

    # rows of extra cells (augment.judged EXT): that sub-bundle's ikc · that bundle's k (merged k − offset)
    sb, sk = AJ.ext_k(Path(bundle), clip, k)
    got = {s: sb / "ikc" / P.ikc_name(clip, f"k{sk:02d}", s) for s in ("right", "left")}
    got = {s: p for s, p in got.items() if p.is_file()}
    miss = [s for s in src.active if s not in got]
    if miss:
        die(input_missing(f"{clip} k={k}: active side {miss} seed missing (judge q) — {sb / 'ikc'}"))
    return got


# ── step list (chain − bake) ──────────────────────────────────────────────
def step_specs(src, shadow: Path, name: str, seed: dict, xr: bool, v: dict) -> list[str]:
    """The same order and argument form as PRE (without bake) [+ idle side] [+ xr] + POST of chain_fast.sh."""
    troot = Path(shadow) / src.trees_rel
    tree = troot / name
    wd = {s: Path(shadow) / src.wrist_rel[s] for s in ("right", "left")}
    ikd = {s: Path(shadow) / src.ik_rel[s] for s in ("right", "left")}
    ikf = {s: ikd[s] / f"{name}.npz" for s in ("right", "left")}

    def wrist(s):
        return f"wrist:--data_dir,{troot},--output_dir,{wd[s]},--side,{s},--sequence,{name}"

    def ik(s):
        return (
            f"ik:--sequence,{name},--tree,{tree},--side,{s},--arm,{src.arm},--hand,{src.hand},"
            f"--base_z_off,{v['zoff']},--pool-root,{v['pool']},--wrist_dir,{wd[s]},--output_dir,{ikd[s]}"
            + (f",--reuse-npz,{seed[s]}" if s in seed else "")
        )

    pre = [f"body:{tree},-q", wrist("left"), wrist("right"), ik("right"), ik("left")]
    if v["idle"]:
        pre += [f"body:{tree},-q"] + [wrist(s) for s in v["idle"]]
    mid = []
    if xr:
        mid = [
            f"xr:--tree,{tree},--arm,{src.arm},--hand,{src.hand},--ik-right,{ikf['right']},--ik-left,{ikf['left']},"
            f"--assets-root,{v['assets']}"
            + (f",--urdf,{os.environ['DEXENGINE_XR_URDF']}" if os.environ.get("DEXENGINE_XR_URDF") else "")
        ]
    post = [
        f"retime:--tree,{tree},--arm,{src.arm},--hand,{src.hand},--ik-right,{ikf['right']},--ik-left,{ikf['left']},"
        f"--safety,{v['safety']},--max-stretch,{v['max_stretch']}",
        f"body:{tree},-q",
        wrist("left"),
        wrist("right"),
    ]
    out = pre + mid + post
    bad = [s for s in out if any("," in str(p) for p in (troot, shadow))]
    if bad:
        die(f"comma in a path breaks the step argument form (comma separated): {troot}")
    return out


# ── resident server pool (the same form as bake_drive; only while this clip generates) ───────────────────────────
def isaac_py(env: dict) -> str:
    if env.get("PYDP"):
        return env["PYDP"]
    r = subprocess.run(
        ["bash", "-c", '. "$0" && isaac_py', str(ROOT / "scripts/_isaac_py.sh")],
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode or not r.stdout.strip():
        die(f"Isaac python not found: {r.stderr.strip()[:200]}")
    return r.stdout.strip()


def alias_base(src, shadow: Path, name: str) -> list[str]:
    """The base of variant V = the base of the original clip (a rigid move does not change the base): the base layer
    entries the bake steps find by tree name (paths.clip_name) (the retiming support stamp coords.stamp_scene -> the
    contact sidecar contact_base/<clip>_<side>.npz · the base tree of that fingerprint paths.ref_base_dir/<clip>) get
    a **relative** symlink with V's name in the shadow (to the original sibling in the same folder; siblings are
    managed by the layer rules augment.scene shadow_dir · relink_shadow); not a copy. Only the bake steps read them
    (env and record read the stamped task_info) · the augment pack does not send symlinks outside variant folders."""
    clip = src.tree_name.removesuffix(P.TREE_SFX)
    vclip = P.clip_name(name)
    m = P.main_root()
    made = []
    cb = P.layer_path("contact_base", src.ds)
    hs = P.hand_spec_dir(src.hand)
    # place = dexengine.core.layout.sidecar_rel: registered clips and their variants V (source = variant_source) share
    # the same
    #   per-hand place, so the link goes there too
    _sc = {(c, s): P.sidecar_file("contact_base", src.ds, c, s, hs) for c in (clip, vclip) for s in ("right", "left")}
    sd = {f.parent for f in _sc.values()}
    if len(sd) != 1:
        die(f"{src.ds}/{clip}: sidecar places differ between sides and variant {vclip} {sorted(map(str, sd))}")
    sd = sd.pop()
    have = [f for (c, _s), f in _sc.items() if c == clip and f.is_file()]
    if not have:
        die(input_missing(f"{src.ds}/{clip}: no original sidecar in {sd}: with 0 aliases the support stamp dies"))
    if sd != cb:
        print(f"[SIDECAR_OWN_ALIAS] {src.ds}/{clip} → {vclip} · {sd.relative_to(m)} · sides {len(have)}", flush=True)
    d = shadow_dir(shadow, m, str(sd.relative_to(m)))
    for f in have:
        if not (d / f.name).exists():
            die(
                input_missing(
                    f"shadow {d} lacks the original sibling {f.name}: the layer disagrees with the base (relink_shadow)"
                )
            )
        t = d / f.name.replace(clip, vclip, 1)
        if not t.is_symlink():
            t.symlink_to(f.name)
        made.append(str(t.relative_to(shadow)))
    rb = P.ref_base_dir(str(src.ti.get("dataset_name")), str(src.ti.get("hand")), str(src.ti.get("embodiment_type")))
    if (rb / clip).is_dir():
        d = shadow_dir(shadow, m, str(rb.relative_to(m)))
        if not (d / clip).exists():
            die(
                input_missing(
                    f"shadow {d} lacks the original sibling {clip}: the layer disagrees with the base (relink_shadow)"
                )
            )
        t = d / vclip
        if not t.is_symlink():
            t.symlink_to(clip)
        made.append(str(t.relative_to(shadow)))
    return made


def bake_env(src, active: str, shadow: Path) -> dict:
    """The same env the bake chain runs in: the canonical shell setup (env after sourcing canon_env.sh, thread counts,
    removed experiment variables) + the axes stage2 passes to the bake (dataset, emb, source layer, judge active side
    mode, hand, arm). Canonical ledger env is removed."""
    if os.environ.get("DEXENGINE_POOL_ROOT"):
        die(
            f"DEXENGINE_POOL_ROOT={os.environ['DEXENGINE_POOL_ROOT']}: the canonical bake clears this value "
            f"(canon_env) · no other pool"
        )
    r = subprocess.run(
        ["bash", "-c", '. "$0" >/dev/null && env -0', str(ROOT / "configs/placement/canon_env.sh")],
        capture_output=True,
        cwd=ROOT,
    )
    if r.returncode:
        die(f"canon_env.sh rc={r.returncode} — {r.stderr.decode(errors='replace')[-200:]}")
    env = dict(x.split("=", 1) for x in r.stdout.decode().split("\0") if "=" in x)
    for k in NO_LEDGER_ENV:
        env.pop(k, None)
    env.update(
        DEXENGINE_DATASET=src.ds,
        DEXENGINE_EMBODIMENT=src.emb,
        DEXENGINE_SRC_LAYER=P.default_src_layer(src.ds),
        DEXENGINE_ACTIVE_SIDES=active,
        DEXENGINE_HAND=src.hand,
        DEXENGINE_ARM=src.arm,
        DEXENGINE_SCENE=bake_scene(src),
        # data root = the shadow (registered; original layers show through symlinks)
        DEXENGINE_MAIN_ROOT=str(shadow),
    )
    return env


def bake_scene(src) -> str:
    """The scene the canonical bake gets: owner of the DEXENGINE_SCENE stage2 exports
    (dexengine.core.datasets.f0_axis(dataset, axis variant)). Without it the bake would use the default desk scene and
    canonical cells of aerial scenes (arctic · show3d) would fail as 'arm guard desk'."""
    from dexengine.data import rules as RR

    return RR.f0_axis(src.ds, RR.ledger_axis_variant(RR.ledger_key(src.ds, src.emb)))["scene"]


class Pool:
    """Starts n resident bake servers (dexengine/placement/bake/server.py, same device) and sends them the step list."""

    def __init__(self, n: int, log_dir: Path, env: dict):
        cpu = env.get("DEXENGINE_BAKE_CPU") == "1"
        if not (cpu or env.get("CUDA_VISIBLE_DEVICES")):
            die(input_missing("no bake GPU pin"))
        self.env = dict(env, CUDA_VISIBLE_DEVICES="") if cpu else env
        self.dev = "cpu" if cpu else env["CUDA_VISIBLE_DEVICES"]
        self.dir = Path(tempfile.mkdtemp(prefix="augbake.", dir=SOCK_BASE))
        self.py = isaac_py(self.env)
        self.procs = []
        t0 = time.time()
        for k in range(n):
            sk = self.dir / f"s{k}.{self.dev}.sock"
            with open(Path(log_dir) / f"server{k}.log", "w") as f:
                argv = [self.py, "dexengine/placement/bake/server.py", "--sock", str(sk), "--ready", f"{sk}.ready"]
                self.procs.append(
                    (sk, subprocess.Popen(argv, cwd=ROOT, env=self.env, stdout=f, stderr=subprocess.STDOUT))
                )
        while not all(Path(f"{sk}.ready").is_file() for sk, _ in self.procs):
            dead = [p.pid for _, p in self.procs if p.poll() is not None]
            if dead or time.time() - t0 > BOOT_S:
                self.stop()
                what = f"dead {dead}" if dead else f"not ready within {BOOT_S}s"
                die(f"resident server {what}: {log_dir}/server*.log")
            time.sleep(0.2)
        self.boot_s = round(time.time() - t0, 1)
        print(
            f"{MARK} pool {self.dir} servers {n} · device {self.dev} · boot {self.boot_s} s · py {self.py}", flush=True
        )

    def run(self, specs: list[str], log: Path) -> int:
        r = subprocess.run(
            [
                sys.executable,
                "dexengine/placement/bake/server.py",
                "send",
                "--pool",
                str(self.dir),
                "--log",
                str(log),
                "--wait",
                str(VARIANT_S),
                "--",
                *specs,
            ],
            cwd=ROOT,
            env=self.env,
            timeout=VARIANT_S * 2,
        )
        return r.returncode

    def stop(self) -> None:
        for sk, p in self.procs:
            if p.poll() is None:
                try:
                    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    c.connect(str(sk))
                    c.sendall(b'{"op": "stop"}\n')
                    c.makefile("rb").readline()
                    c.close()
                    p.wait(timeout=30)
                except (OSError, subprocess.TimeoutExpired):
                    p.kill()
                    p.wait()
        shutil.rmtree(self.dir, ignore_errors=True)
