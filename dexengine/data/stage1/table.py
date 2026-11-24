"""The stage-1 step table: what each step runs and how its progress is counted.

Each row of ``STAGES`` is ``(name, cell, interpreter slot, command)``. The name's first token (``2``, ``2b`` ...) is
the step id. The command is a ``Run`` (the argv this package drives) or a shell line printed as guidance. The
interpreter slot names the interpreter: ``retarget`` = the retargeting venv (``paths.retarget_py``), ``""`` = a shell
entry point that picks its own.

Cells (how a step's progress is counted):

- ``_COVER``: parent takes retargeted with the current MANO frame rule / parent takes the base layer needs (step 1 is
  counted in takes, every later step in segments).
- ``_KEYED + text``: trees whose ``task_info.json`` contains ``text`` (the step's own marker). With ``Run.every`` every
  base tree must carry it (cell x/y); otherwise any tree counts (cell = count, 0 = empty).
- ``_ASSIGN_KEYED``: like ``_KEYED`` with the hand's case-assignment version as the text.
- ``_INPUT_FP + module``: trees whose input fingerprint equals the current inputs / all trees; the judgement is the
  producer module's ``freshness(ds, spec_dir)`` (and ``tree_fresh(ds, spec_dir, tree)`` per tree).
- ``_SIDECAR_FP + layer``: (clip, side) contact sidecars whose fingerprint matches this hand's base / expected.
- ``_F0_JUDGED``: clips the frame-0 pre-judge has settled / clips of the base layer.
- ``_POOL`` / ``_UNI`` / ``_TREES``: IK pool sides, one-hand scene templates, baked trees (per combo, not per dataset).

``Run`` fields: ``axis`` ("" once per dataset, "emb" per embodiment, "emb_side" per embodiment and arm side);
``argv`` with slots ``{ds} {hand} {emb} {side}`` and path tokens ``@base @wrist @clips @f0`` (resolved by
``paths.layer_path`` and the f0 axis); ``rule`` = a rule id whose dataset scope is the step's scope (outside it the step
is "not applicable", shown as ``-``); ``scene`` = prefix the dataset's judge scene (``DEXENGINE_SCENE``) for steps that
read the desk; ``reuse`` = a per-tree reuse command sent instead of ``argv`` when the caller says the rerun has the same
identity; ``every`` = see ``_KEYED``; ``reads`` / ``writes`` = the ARTIFACTS the step opens / makes besides its tree
(``check_inputs``); ``cut_copies`` = the task_info ``rules`` key the step stamps when the long-clip cut copies that
stamp onto each piece (``segment.pieces.PIECE_RULES_COPY``; ``split_steps``).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

from dexengine.data import rules as RR
from dexengine.data.artic.convert_version import D1G_VERSION as _D1G_VERSION
from dexengine.data.cases import assign_version as _AR
from dexengine.data.desk_penetration import D24_VERSION as _D24_VERSION

REPO = Path(__file__).resolve().parents[3]

_COVER = "@parent-take-coverage"
_KEYED = "@task-info-contains:"
_ASSIGN_SLOT = "{assign_version_of_hand}"
_ASSIGN_KEYED = _KEYED + _ASSIGN_SLOT
_TREES = "@trees-layer"
_POOL = "@ikpool"
_UNI = "@uni-template"
_SIDECAR_FP = "@sidecar-fp:"
_INPUT_FP = "@input-fp:"
_F0_JUDGED = "@f0-judged"

Run = namedtuple(
    "Run", "axis argv rule scene reuse every reads writes cut_copies", defaults=(None, False, (), False, (), (), "")
)

#: What a step reads or writes besides the tree it walks: the two contact sidecar layers (named as paths.sidecar_file
#: names them), the D1-g merge stamp (task_info rules.D1g), the case stamp (task_info case: contact_frac · grasp_s ·
#: fps), the hand-object assignment record (rules.ASSIGN), the applied roles (D23: main objects, sides, supports) and
#: the support-gap stamp (rules[support.gap.RULE_KEY]). Each has one writer; a reader must come after it
#: (check_inputs). Where each lives on a tree: stage1.artifacts.PLACES (a per-tree input key strips the fields of
#: artifacts written after its step).
ARTIFACTS = ("d1g", "contact_all", "case", "assign", "roles", "contact_base", "support_gap")
#: The sidecar layers among them — the ones the run guard (paths.sidecar_file under DEXENGINE_STAGE1_STEP) measures.
SIDECARS = ("contact_all", "contact_base")

STAGES = (
    (
        "1 retarget",
        _COVER,
        "",
        "python scripts/stage1_retarget.py --arm <arm> --hand <hand> --dataset taco --emb bimanual"
        " --bases <list>   # TACO · other datasets: python3 -m dexengine.retarget.reretarget <ds>",
    ),
    # The base layer is current when every tree records the current adapter version (base_layer.stale.freshness).
    (
        "2 base layer",
        _INPUT_FP + "dexengine.data.base_layer.stale",
        "retarget",
        Run(
            "",
            ("dexengine/data/base_adapter.py", "--dataset", "{ds}", "--hand", "{hand}"),
            reuse=("dexengine/data/base_adapter.py", "--dataset", "{ds}", "--hand", "{hand}", "--skip-existing"),
        ),
    ),
    # 2b and 2c are dataset-specific base-layer rules. They run after the base layer and before cases (3): D1-g changes
    # the object list (a part pair becomes one articulated object) and 3, 4, 4b and 7 read that list. Both are
    # idempotent. 2b first builds or checks the articulation spec (artic.spec_stage.ensure) so that more hands never
    # move the joint ranges; it uses a GPU (first device of DEXENGINE_F0_GPUS) when assets must be baked.
    (
        "2b articulated D1-g",
        _KEYED + json.dumps(_D1G_VERSION),
        "retarget",
        Run(
            "emb",
            ("dexengine/data/oakink2/artic_convert.py", "--emb", "{emb}", "--write"),
            rule="D1-g",
            writes=("d1g",),
            cut_copies="D1g",
        ),
    ),
    (
        "2c joint value D25",
        _KEYED + '"D25"',
        "retarget",
        Run("emb", ("dexengine/data/artic/arctic_align.py", "--emb", "{emb}", "--write"), rule="D25", cut_copies="D25"),
    ),
    # 2d: the all-object contact sidecar read by cases (3), checked against this hand's base and rebuilt when it
    # differs. The layer is hand independent but built from one hand's object poses.
    (
        "2d contact sidecar (all objects)",
        _SIDECAR_FP + "contact_all",
        "retarget",
        Run(
            "",
            ("dexengine/data/contact/sidecar.py", "--dataset", "{ds}", "--hand", "{hand}", "--all-objects"),
            writes=("contact_all",),
        ),
    ),
    # 2e: wrist tilt repair of references whose wrist housing enters the desk (placement.repair.wrist_desk_stage).
    # After 2d (it reads contact_all), before 3 (cases, roles, quality, wrist cache and f0 read the repaired hand).
    (
        "2e wrist tilt",
        _INPUT_FP + "dexengine.placement.repair.wrist_desk_stage",
        "retarget",
        Run(
            "",
            ("dexengine/placement/repair/wrist_desk_stage.py", "--dataset", "{ds}", "--hand", "{hand}", "--write"),
            rule="WRIST_SLACK",
            scene=True,
            reads=("contact_all",),
            cut_copies="WRIST_SLACK",
        ),
    ),
    # 3 and 4 write into the base trees; they count by their own marker on every tree (the base adapter rewrites a
    # tree folder whole and the long-clip split drops `case`/`rules`, so a rerun upstream clears the markers).
    # 4 measures D24 on each main object's USD: the object USD bake runs before it, and a main object without a USD
    # file fails the step (rc 4, the bake printed per object).
    (
        "3 cases",
        _ASSIGN_KEYED,
        "retarget",
        Run(
            "",
            ("dexengine/data/cases/taxonomy.py", "--dataset", "{ds}", "--write"),
            every=True,
            reads=("contact_all",),
            writes=("case", "assign"),
        ),
    ),
    (
        "4 roles D23",
        _KEYED + json.dumps(_D24_VERSION),
        "retarget",
        Run(
            "",
            ("dexengine/data/object_roles.py", "--datasets", "{ds}", "--write"),
            every=True,
            reads=("assign",),
            writes=("roles",),
        ),
    ),
    # 4b: quality flags and the training-candidate flag (filtering = dexengine.data.filtering). After roles (it reads
    # the main object), before f0 (its input key reads the task_info written here). Counted by input fingerprint.
    (
        "4b quality D12-c",
        _INPUT_FP + "dexengine.data.filtering",
        "retarget",
        Run("", ("dexengine/data/filtering.py", "--dataset", "{ds}", "--hand", "{hand}"), reads=("assign", "roles")),
    ),
    # 5: the main-object contact sidecar (contact_base) read by the bake and D26. After roles (4).
    (
        "5 contact sidecar (main object)",
        _SIDECAR_FP + "contact_base",
        "retarget",
        Run(
            "",
            ("dexengine/data/contact/sidecar.py", "--dataset", "{ds}", "--hand", "{hand}"),
            reads=("roles",),
            writes=("contact_base",),
        ),
    ),
    # 2f: S-hand-clear — a reference hand inside a still extra is measured only: the side keeps its reference as
    # recorded (KEPT_AS_RECORDED, with its max depth; no tree is written); a true-shape object overlap or a float jump
    # drops the clip for this hand — placement.repair.hand_clear_stage.
    # It reads the D1-g stamp (2b), the all-object sidecar (2d), the case stamp (3: the extra classifier's human
    # contact), the main objects (4) and their contact sidecar (5: the float rule's first touch and start grip), so it
    # runs after 5 — one pass from an empty root (check_inputs). It writes no tree (measured only), so 3, 4 and f0 read
    # the same trees either side of it; its ledger and exclusion lines are read after stage 1 (candidates). Its input
    # tree is still the one 2e left (3 … 5 write no qpos). The id keeps the rule's name; the row order is the run
    # order.
    (
        "2f hand clear",
        _INPUT_FP + "dexengine.placement.repair.hand_clear_stage",
        "retarget",
        Run(
            "",
            ("dexengine/placement/repair/hand_clear_stage.py", "--dataset", "{ds}", "--hand", "{hand}", "--write"),
            rule="S-hand-clear",
            reads=("d1g", "contact_all", "case", "roles", "contact_base"),
        ),
    ),
    # 5b: the support gap of aerial-scene trees, measured once and stamped (the candidate filter only reads the stamp).
    (
        "5b support gap",
        _INPUT_FP + "dexengine.data.support.gap",
        "retarget",
        Run(
            "",
            ("dexengine/data/support/gap.py", "--dataset", "{ds}", "--hand", "{hand}", "--write"),
            reads=("roles", "contact_base"),
            writes=("support_gap",),
        ),
    ),
    # 6: the wrist pose cache, once per arm side, from each tree's own body_positions. `--skip_fresh` rewrites only
    # caches whose value changed (unchanged bytes keep the f0 key).
    (
        "6 wrist cache",
        _INPUT_FP + "dexengine.placement.fk.wrist_poses",
        "retarget",
        Run(
            "emb_side",
            (
                "dexengine/placement/fk/wrist_poses.py",
                "--data_dir",
                "@base",
                "--output_dir",
                "@wrist",
                "--side",
                "{side}",
                "--skip_fresh",
            ),
        ),
    ),
    # 7: the frame-0 pre-judge. Its clip list is counted from the base layer (one name per line; names contain
    # spaces). Write location and scene come from the same function as its cell (rules.f0_axis).
    (
        "7 f0 shadow",
        _F0_JUDGED,
        "retarget",
        Run(
            "emb",
            (
                "dexengine/retarget/f0/prejudge.py",
                "--hand",
                "{hand}",
                "--dataset",
                "{ds}",
                "--emb",
                "{emb}",
                "--clips",
                "@clips",
                "--f0-root",
                "@f0",
            ),
            scene=True,
            reads=("roles",),
        ),
    ),
    # 8 and 8b are per combo (arm + hand), the same for every dataset (marked †).
    (
        "8 IK pool†",
        _POOL,
        "",
        "bash scripts/build_reach_map.sh <arm> <hand> [GPU] [workers]   # ~94 MB per side · once per combo · "
        "when the pool"
        " exists and only the clearance sidecar's ruler/shape differs: python3 "
        "dexengine/placement/reach_map/clearance.py"
        " --arm <arm> --hand <hand> --force",
    ),
    (
        "8b one-hand scene templates†",
        _UNI,
        "",
        "python3 dexengine/placement/uni_layout.py --extract-for <hand>   # once per hand · after the "
        "stage-1 base layer",
    ),
    (
        "9 trees (judge · bake)",
        _TREES,
        "",
        "bash scripts/stage2_ground.sh --arm <arm> --hand <hand> --dataset <ds> --emb <emb> --clips <list> --fold "
        "$(python3 -m dexengine.data.rules --launch-flags <ds>)",
    ),
)


def check_inputs(stages=None) -> dict:
    """The table check (dies on the first defect): every declared read or write names an ARTIFACTS entry, each artifact
    has exactly one writer, and every read comes after its writer in row order — a step that reads what a later step
    makes cannot run once from an empty root. Returns {step id: artifacts written at or before that step}."""
    stages = STAGES if stages is None else stages
    writer, made, out = {}, set(), {}
    for g in stages:
        cmd = g[3]
        reads, writes = (cmd.reads, cmd.writes) if isinstance(cmd, Run) else ((), ())
        for a in (*reads, *writes):
            if a not in ARTIFACTS:
                raise SystemExit(f"[STAGE1_ORDER] step {step_id(g)}: unknown artifact {a!r} (ARTIFACTS {ARTIFACTS})")
        for a in reads:
            if a not in made:
                later = next((step_id(h) for h in stages if a in getattr(h[3], "writes", ())), None)
                raise SystemExit(
                    f"[STAGE1_ORDER] step {step_id(g)} reads {a!r}, which no earlier step writes "
                    f"(writer: {later or 'none'}) — move the step after its writer"
                )
        for a in writes:
            if a in writer:
                raise SystemExit(f"[STAGE1_ORDER] {a!r} has two writers: {writer[a]} and {step_id(g)}")
            writer[a] = step_id(g)
        made |= set(writes)
        out[step_id(g)] = frozenset(made)
    return out


def driven_steps(stages=None) -> list[str]:
    """Ids of the steps `--emit-run` drives (Run rows), in run order — the one list the stage-1 driver walks."""
    return [step_id(g) for g in (STAGES if stages is None else stages) if isinstance(g[3], Run)]


def sidecars_ready(step: str) -> list[str]:
    """The contact sidecar layers written at or before `step` — what that step may open (the run guard's list)."""
    return [a for a in SIDECARS if a in _MADE[step]]


#: Interpreter slot -> `paths --what` value.
_PY_SLOT = {"retarget": "retarget-py"}

#: Path tokens as the printed guidance shows them (a person can paste the command).
_DOC_TOKEN = {
    "@base": "$(python3 dexengine/core/paths.py --what layer --dataset <ds> --hand <hand> --emb <emb> --layer base)",
    "@wrist": "$(python3 dexengine/core/paths.py --what layer --dataset <ds> --hand <hand> --emb <emb>"
    " --layer wrist_base)/wrist_poses_<side>",
    "@clips": "<base-layer clip list file -- written by --emit-run>",
    "@f0": "$(python3 -m dexengine.data.rules --f0-axis <ds> | cut -f2,3 --output-delimiter=/)",
}
_DOC_SCENE = "DEXENGINE_SCENE=$(python3 -m dexengine.data.rules --f0-axis <ds> | cut -f1) "
_DOC_AXIS = {"": "", "emb": "       # once per emb", "emb_side": "       # once per emb x side (right, left)"}


def step_id(stage) -> str:
    """The step id of a STAGES row (first token of its name)."""
    return stage[0].split()[0]


def py(what: str) -> str:
    """The interpreter `paths --what <what>` names; an empty answer is fatal (the command would be malformed)."""
    r = subprocess.run(
        [sys.executable, str(REPO / "dexengine/core/paths.py"), "--what", what],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise SystemExit(
            f"[hand_pipeline] paths.py --what {what} failed (rc={r.returncode}): "
            f"{(r.stderr or r.stdout).strip().splitlines()[-1:] or ['(no output)']}"
        )
    return r.stdout.strip()


def cell_needle(pat: str, hand_dir: str) -> bytes:
    """The bytes a `_KEYED` cell looks for; the assignment-version slot becomes that hand's version."""
    needle = pat[len(_KEYED) :]
    return (json.dumps(_AR.hand_version(hand_dir)) if needle == _ASSIGN_SLOT else needle).encode()


def rule_scope(cmd) -> tuple:
    """The datasets a step applies to (empty = all). A rule id missing from the registry is fatal."""
    rid = getattr(cmd, "rule", None)
    if not rid:
        return ()
    hit = [r for r in RR.RULES if r.id == rid]
    if len(hit) != 1:
        raise SystemExit(f"[hand_pipeline] step rule {rid!r} appears {len(hit)} times in the rule registry")
    return tuple(hit[0].datasets)


def applies(cmd, ds: str) -> bool:
    """Does the step apply to dataset `ds` (its rule's scope)?"""
    sc = rule_scope(cmd)
    return not sc or ds in sc


def _has_tree_fresh(module: str) -> bool:
    """Does `module` define a top-level ``tree_fresh`` (read from its source: the producers pull in numpy/mujoco)?"""
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        raise SystemExit(f"[STAGE1_ORDER] input-fingerprint module {module!r} is not importable")
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    return any(isinstance(n, ast.FunctionDef) and n.name == "tree_fresh" for n in tree.body)


def split_steps(stages=None) -> dict[str, list[str]]:
    """What the long-clip split (stage1.split_apply) does with each driven step after the base layer, in row order.
    The cut makes new trees of the layer the first driven step builds, so every later step must hold for them:

    - ``carried``: the step stamps a task_info ``rules`` key that the cut copies onto every piece (``Run.cut_copies``,
      which must be in ``segment.pieces.PIECE_RULES_COPY``); the piece is a window of the parent arrays it fixed.
    - ``run``: the step's cell judges one tree (an input fingerprint whose module has ``tree_fresh``, an every-tree
      marker, a contact sidecar); split_apply reruns it on the pieces ``segment.todo`` reports, and that judgement
      reads the same list.
    - ``f0``: the frame-0 pre-judge (``_F0_JUDGED``); the feeder's axis f0 runs it after the split.

    A step in none of them dies, and so does a ``cut_copies`` key the cut does not copy."""
    from dexengine.data.segment.pieces import PIECE_RULES_COPY  # noqa: PLC0415

    driven = [g for g in (STAGES if stages is None else stages) if isinstance(g[3], Run)]
    out: dict[str, list[str]] = {"carried": [], "run": [], "f0": []}
    for g in driven[1:]:
        pat, cmd, sid = g[1], g[3], step_id(g)
        if cmd.cut_copies:
            if cmd.cut_copies not in PIECE_RULES_COPY:
                raise SystemExit(
                    f"[STAGE1_ORDER] step {sid}: cut_copies {cmd.cut_copies!r} is not a key the long-clip cut copies "
                    f"(segment.pieces.PIECE_RULES_COPY {PIECE_RULES_COPY}) — new pieces would miss the step"
                )
            out["carried"].append(sid)
        elif pat == _F0_JUDGED:
            out["f0"].append(sid)
        elif (
            pat.startswith(_SIDECAR_FP)
            or (pat.startswith(_KEYED) and every_tree(pat, cmd))
            or (pat.startswith(_INPUT_FP) and _has_tree_fresh(pat[len(_INPUT_FP) :]))
        ):
            out["run"].append(sid)
        else:
            raise SystemExit(
                f"[STAGE1_ORDER] step {sid}: the long-clip split neither carries it to new pieces (Run.cut_copies), "
                f"nor judges it per tree to rerun it (cell {pat!r}), nor hands it to f0 — new pieces would miss it"
            )
    return out


def every_tree(pat: str, cmd) -> bool:
    """Must the step's marker be on every base tree (``Run.every``)? Only `_KEYED` cells may say so."""
    ev = bool(getattr(cmd, "every", False))
    if ev and not pat.startswith(_KEYED):
        raise SystemExit(f"[hand_pipeline] Run.every is only for `{_KEYED}` cells -- cell {pat!r}")
    return ev


def tree_unit(pat: str, cmd) -> bool:
    """Is the step's cell counted per base tree (input-fingerprint cells and every-tree marker cells)?"""
    return pat.startswith(_INPUT_FP) or every_tree(pat, cmd)


def embs(ds: str) -> tuple[str, ...]:
    """The dataset's embodiments (rule-registry profile); an unknown dataset raises KeyError."""
    return RR.DATASET_PROFILE[ds].embs


def emb_sides(emb: str) -> tuple[str, ...]:
    """The arm sides an embodiment uses: bimanual = both, right/left = that side."""
    return ("right", "left") if emb == "bimanual" else (emb,)


def cmd_of(stage) -> str:
    """A command line for the step that can be pasted into a shell (generated from the Run argv)."""
    slot, cmd = stage[2], stage[3]
    if isinstance(cmd, Run):
        body = " ".join(_DOC_TOKEN.get(a, a) for a in cmd.argv)
        for k, v in (("{ds}", "<ds>"), ("{hand}", "<hand>"), ("{emb}", "<emb>"), ("{side}", "<side>")):
            body = body.replace(k, v)
        env = _DOC_SCENE if cmd.scene else ""
        return env + (f"{py(_PY_SLOT[slot])} {body}" if slot else body) + _DOC_AXIS[cmd.axis]
    if not slot:
        return cmd
    return f"{py(_PY_SLOT[slot])} {cmd}"


#: The table check runs at import: a table whose order breaks a declared input never loads.
_MADE = check_inputs()
