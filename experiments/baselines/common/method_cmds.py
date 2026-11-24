"""Per-clip method command rows of the Tab. 2 baseline queues (used by mkcmds.py and the cluster job table): the
clip's line from ``<workspace>/run/*_cmds.txt`` with only the variant's inputs and outputs swapped (each swap asserted
to hit the expected number of places), plus the object-mass run-time gate."""

from __future__ import annotations

import re
import sys

from experiments import workspace
from experiments.baselines.common import prep_mass_0p2
from experiments.baselines.maniptrans import agg_override

METHODS = ("spider", "egoengine", "chord", "maniptrans")


# Object-mass sets: label -> the tool that builds the derived scene set data/<set>_<label> and checks it at run time.
# The row runs the method only after that check passes on the scenes it loads ('[MASS_0P2] … ok' in the method log).
MASS_SETS = {"m0p2": workspace.REPO / "experiments/baselines/common/prep_mass_0p2.py"}


SPIDER_STARTCHECK = {"on": "start_a", "off": "start_a_nocheck"}  # switch value -> output suffix (never the same files)


def fail(msg: str) -> None:
    sys.exit(f"[MKCMDS] FAIL {msg}")


def root() -> str:
    """The workspace root as it appears in the generated command lines."""
    return str(workspace.root())


def mk_dirs() -> str:
    B = root()
    return "mkdir -p " + " ".join(
        f"{B}/{d}/{m}" for d in ("logs", "results", "logs_short", "results_short") for m in METHODS
    )


def line_for(f: str, clip: str) -> str:
    """The one line of run/<f> that belongs to <clip>."""
    rows = [ln for ln in (workspace.root() / "run" / f).read_text().splitlines() if ln and not ln.startswith("#")]
    hit = [
        ln
        for ln in rows
        if (f"task={clip} " in ln)
        or (f"--task {clip} " in ln)
        or (f"/results/chord/{clip} " in ln)
        or (f"/results/maniptrans/{clip}.npz" in ln)
    ]
    if len(hit) != 1:
        fail(f"{f}: {len(hit)} lines for clip {clip}")
    return hit[0]


def sub(s: str, old: str, new: str, n: int = 1) -> str:
    c = s.count(old)
    if c != n:
        fail(f"swap {old!r}: found {c}, want {n}")
    return s.replace(old, new)


def rsub(s: str, pat: str, new: str, n: int = 1) -> str:
    out, c = re.subn(pat, new, s)
    if c != n:
        fail(f"regex {pat!r}: found {c}, want {n}")
    return out


def mass_gate(line: str, py: str, root_name: str, mass: str, ac: str, lg: str, tail: str) -> str:
    """Prefix the method call of a production line with the derived set's mass check (same log; the method appends)."""
    B = root()
    line = sub(
        line,
        "&& CUDA_VISIBLE_DEVICES=",
        f"&& {py} -I {MASS_SETS[mass]} --check {B}/data/{root_name} {ac} > {lg} 2>&1 && CUDA_VISIBLE_DEVICES=",
    )
    return sub(line, f" > {lg} 2>&1; {tail}", f" >> {lg} 2>&1; {tail}")


def spider_a_cmd(ac: str, startcheck: str | None = None, mass: str | None = None) -> str:
    """SPIDER production line on the start-state (a) inputs (data/spider_a), stale rollout removed first.

    startcheck: None = the line as is (SPIDER's own start-state check on by its default, no marker); 'on' / 'off' =
    explicit, where 'off' adds SPIDER's override +sanity_check_seconds=0.0. Both explicit forms write
    `[SPIDER_STARTCHECK] <on|off> rc=<rc> applied <the value SPIDER saved in its config.yaml>` to the method log and use
    their own output suffix. mass = an object-mass set (MASS_SETS; needs the explicit switch)."""
    B = root()
    sfx = "start_a" if startcheck is None else SPIDER_STARTCHECK[startcheck]
    if mass is not None and (mass not in MASS_SETS or startcheck is None):
        fail(f"mass {mass!r}: a known set with the explicit start-check switch")
    rname = "spider_a" if mass is None else f"spider_a_{mass}"
    sfx += "" if mass is None else f"_{mass}"
    sa = line_for("spider_cmds.txt", ac).replace(f"dataset_dir={B}/data/spider ", f"dataset_dir={B}/data/{rname} ")
    sa = sa.replace(f"{B}/data/spider/processed/", f"{B}/data/{rname}/processed/")
    sa = sa.replace(f"{B}/logs/spider/{ac}.log", f"{B}/logs/spider/{ac}.{sfx}.log")
    sa = sa.replace(f"{B}/results/spider/{ac}.json", f"{B}/results/spider/{ac}.{sfx}.json")
    if f"{B}/data/spider/" in sa or sfx not in sa:
        fail("spider (a) line still reads the staged inputs")
    od = f"{B}/data/{rname}/processed/taco/sharpa_wave/bimanual/{ac}/0"
    pre = f"{mk_dirs()}; rm -f {od}/trajectory_mjwp.npz"
    if startcheck is not None:
        if startcheck == "off":
            # '+': the key is not in SPIDER's hydra default.yaml (a plain override is refused); filter_config_fields
            # keeps it
            sa = sub(sa, f"dataset_dir={B}/data/{rname} ", f"dataset_dir={B}/data/{rname} +sanity_check_seconds=0.0 ")
        lg = f"{B}/logs/spider/{ac}.{sfx}.log"
        if mass is not None:
            sa = mass_gate(sa, ".venv/bin/python", rname, mass, ac, lg, "echo rc=$? >>")
        sa = sub(
            sa,
            f"; echo rc=$? >> {lg};",
            f'; rc=$?; echo rc=$rc >> {lg}; echo "[SPIDER_STARTCHECK] {startcheck} rc=$rc applied '
            f"$(grep -h '^sanity_check_seconds:' {od}/config.yaml || echo NONE)\" >> {lg};",
        )
        pre += f" {od}/config.yaml"
    return f"{pre}; {sa}"


def egoengine_a_cmd(ac: str, mass: str | None = None, run: str | None = None) -> str:
    """EgoEngine production line on the (a) inputs (data/egoengine_a), inner cap 6900 s under the row cap 7200 s.

    mass = an object-mass set (data/egoengine_a_<mass>, suffix start_a_<mass>, the set's check first). run = r1 | r2: a
    further EgoEngine run in the same tree, with its own copy of the scene set (data/egoengine_a_<mass>_<run>) and
    output
    suffix, because the method writes its trajectory into the dataset dir."""
    B = root()
    if (mass is not None and mass not in MASS_SETS) or (
        run is not None and (mass is None or run not in ("r1", "r2", "r3"))
    ):
        fail(f"mass {mass!r} run {run!r}")
    tail_ = "" if run is None else f"_{run}"
    rname = "egoengine_a" if mass is None else f"egoengine_a_{mass}{tail_}"
    sfx = "start_a" if mass is None else f"start_a_{mass}{tail_}"
    ea = line_for("egoengine_cmds.txt", ac).replace(
        f"--dataset-dir {B}/data/egoengine ", f"--dataset-dir {B}/data/{rname} "
    )
    ea = ea.replace(f"{B}/data/egoengine/processed/", f"{B}/data/{rname}/processed/")
    ea = ea.replace(f"{B}/logs/egoengine/{ac}.log", f"{B}/logs/egoengine/{ac}.{sfx}.log")
    ea = ea.replace(f"{B}/results/egoengine/{ac}.json", f"{B}/results/egoengine/{ac}.{sfx}.json").replace(
        "timeout 10800", "timeout 6900"
    )
    if f"{B}/data/egoengine/" in ea or sfx not in ea:
        fail("egoengine (a) line still reads the staged inputs")
    if mass is not None:
        ea = mass_gate(
            ea, ".venv-action/bin/python", rname, mass, ac, f"{B}/logs/egoengine/{ac}.{sfx}.log", "echo rc=$? >>"
        )
    return (
        f"{mk_dirs()}; rm -f "
        f"{B}/data/{rname}/processed/taco/sharpa_wave/bimanual/{ac}/0/trajectory_mjwp_mixed.npz; {ea}"
    )


def mt_agg(cmd: str, c: str) -> str:
    """Apply the aggregateMode=0 override (maniptrans/agg_override.py owns the clip list) to a ManipTrans residual
    row.
    """
    return agg_override.rewrite(cmd, c) if c in agg_override.ROWS else cmd


def maniptrans_clip_cmd(c: str) -> str:
    """ManipTrans production line: retarget x2 -> residual (4096 envs, early stop >= 100 epochs, 2 h budget) -> recorder
    (64 worlds) -> score, stale outputs removed first."""
    B = root()
    ln = line_for("maniptrans_cmds.txt", c)
    return mt_agg(f"{mk_dirs()}; rm -f {B}/results/maniptrans/{c}.npz {B}/results/maniptrans/{c}.json; {ln}", c)


def owner_mass() -> float:
    """Our fixed object mass, read from its owner by prep_mass_0p2 (which also ties the value to the set label m0p2)."""
    M = prep_mass_0p2.owner_mass()
    if f"m{M:g}".replace(".", "p") not in MASS_SETS:
        fail(f"owner DEFAULT_OBJECT_MASS_KG {M} is not a MASS_SETS label")
    return M


def maniptrans_m0p2_cmd(c: str) -> str:
    """The ManipTrans production line at our fixed object mass: E123_OBJ_MASS_KG exported (read by the ManipTrans port,
    dexhandmanip_bih.py: mass M, inertia of the same hulls scaled by M / m_shape, logged as [MT_MASS]) and its own
    outputs
    (.m0p2 results/logs, experiment e123_<clip>_m0p2, so the recorder's checkpoint glob never picks the plain run).
    The retargets are kinematic and run as in the plain line."""
    B = root()
    ln = line_for("maniptrans_cmds.txt", c)
    M = owner_mass()
    ln = sub(ln, f"{B}/results/maniptrans/{c}.npz", f"{B}/results/maniptrans/{c}.m0p2.npz", 2)
    ln = sub(ln, f"{B}/results/maniptrans/{c}.json", f"{B}/results/maniptrans/{c}.m0p2.json")
    ln = sub(ln, f"{B}/logs/maniptrans/{c}.", f"{B}/logs/maniptrans/{c}.m0p2.", 4)
    ln = sub(ln, f" experiment=e123_{c} ", f" experiment=e123_{c}_m0p2 ")
    ln = sub(ln, f"/runs/e123_{c}__*/nn/e123_{c}.pth", f"/runs/e123_{c}_m0p2__*/nn/e123_{c}_m0p2.pth")
    ln = sub(ln, "&& export PATH=", f"&& export E123_OBJ_MASS_KG={M!r} PATH=")
    return mt_agg(
        f"{mk_dirs()}; rm -f {B}/results/maniptrans/{c}.m0p2.npz {B}/results/maniptrans/{c}.m0p2.json; {ln}", c
    )


def chord_cmd(c: str) -> str:
    """FlashCHORD production line verbatim (train, eval, score), with CHORD's own start scene."""
    return f"{mk_dirs()}; " + line_for("chord_cmds.txt", c)
