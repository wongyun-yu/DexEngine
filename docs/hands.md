# Hands

A hand is a spec file plus the assets built from it. Every hand-dependent constant of the pipeline lives in its
`HandSpec`, read from `configs/robots/hands/<hand>.yaml` by the one loader `dexengine.core.embodiment`. The code reads
hand facts from the registries; it never matches on hand names.

| Hand id | Hand | Spec file | Retargeting spec directory |
|---|---|---|---|
| `sharpa_wave` | Sharpa Wave | `configs/robots/hands/sharpa_wave.yaml` | `sharpa_wave` |
| `wuji2` | Wuji Hand 2 | `configs/robots/hands/wuji2.yaml` | `wuji2` |
| `xhand_pro` | ROBOTERA XHand Pro | `configs/robots/hands/xhand_pro.yaml` | `xhand_pro` |
| `dg5f_s` | Tesollo DG-5F-S | `configs/robots/hands/dg5f_s.yaml` | `dg5f_s` |

A spec file states every field of its dataclass. The loader stops on an unknown key, a missing key or a value of the
wrong type, naming the file and the key, and prints `[EMBODIMENT] hand=<h> arm=<a> mount=<a>+<h> spec_sha=<16 hex>`
when a combo is resolved (`spec_sha` = sha256 of the three files).

`python -m dexengine.core.paths --what hand-spec-dir --hand <hand>` prints the spec directory, and `--what combo-ids`
prints the arm × hand combos.

## What registers a hand

| Entry | Where | What it holds |
|---|---|---|
| `HandSpec` | `configs/robots/hands/<hand>.yaml` → `dexengine.core.hands.HAND_REGISTRY` | Joints in action order, coupled joints, qpos width, tip/palm/contact bodies, vendor URDF and meshes, the merge subtree root (`urdf_subtree_root`), the mounting face → root distance (`mount_face_to_root`), finger effort and velocity caps, PD gains, fingertip pad offsets, the spec directory (`retarget_dirname`) |
| `MountSpec` | `configs/robots/mounts/<arm>+<hand>.yaml` → `dexengine.core.mounts.MOUNT_REGISTRY[(arm, hand)]` | End-effector → hand base translation and per-side rotation, its source, a `provisional` flag, render-only parts (adapter plates) |
| Retargeting asset | `dexengine.core.layout.RETARGET_HAND_ASSETS[<spec dir>]` | sha256 of `right.xml ‖ left.xml ‖ bimanual.xml`, joint count, status, hand id, provenance |
| RL recipe | the hand file's `rl_recipe` → `dexengine.core.hands.HAND_RL_RECIPES[<hand>]` | Hand-specific training settings (`null` for none; every hand file states it) |
| Render look | `dexengine/core/hand_looks.py` | Material rules that give the render body the manufacturer's appearance |

Hardware facts have no defaults. A `HandSpec` without its vendor effort and velocity caps or its finger gains fails at
construction. `finger_gains_from_cap` derives gains from the caps by the rule in `dexengine/core/hands/spec.py`. A
mount marked `provisional` is printed with a marker on every run that uses it.

The arm side is `configs/robots/arms/<arm>.yaml` → `dexengine.core.arms.ARM_REGISTRY` (`ArmSpec`: joint classes,
vendor limits, end-effector body, the tool flange, the skeleton the merge cuts from the vendor URDF, whether it has a
torso to guard, the link groups placement reads, the idle-arm rest pose, the default arm and head drives with the head
pitch and the limb joints of the judge's natural posture — `null` where the arm has none). The full-robot USD path is a pair fact (`paths.combo_usd_relpath`).

**The mount is three parts.** The arm states its flange (`flange.xyz`, `flange.approach` in the end-effector frame),
the hand states the distance from its mounting face to its subtree root (`mount_face_to_root`), and the mount file
states the adapter between them. With `translation: {stack: {adapter_thickness: t}}` the loader composes the offset
`flange.xyz + flange.approach · (t + mount_face_to_root)` on both sides; it checks that the mount rotation maps the
hand root's +z onto the approach. A transform read from a vendor document is stated as
`translation: {measured: {right: [x, y, z], left: [x, y, z]}}` instead.
Body generations, that is which merged URDF and USD variant a combo uses for simulation and for rendering, are
stated in the arm file (`generations`: the canonical body and its render plate; `collider_rev`: the sim collider
revision); `dexengine.core.arm_generations` derives the other generations from them (revision on top, hand joint
limit variants). The asset folder of a combo is `dexengine.core.paths.combo_dirname(<combo>)`.

## Assets of a hand

| Asset | Location | Built by |
|---|---|---|
| Vendor URDF + meshes (+ MJCF) | `HandSpec.vendor_urdf` / `vendor_meshes` (/ `vendor_mjcf`): `assets/robots/hands/sharpa_wave/{side}_sharpa_wave/` for Sharpa Wave, `<data root>/data/_stage/<hand>/` for the others; none in git | `python -m dexengine.robots.vendor fetch` (sources and sha256: `configs/robots/vendor/`) |
| Retargeting rig (MJCF) | `assets/robots/hands/<spec>/mjcf/{right,left,bimanual}.xml`, mesh `assets/` (built, not in git), `retarget_prior.yaml` (in git) | `<retarget venv> dexengine/robots/hand_rig.py <hand>` (all hands: `python -m dexengine.robots.vendor build`) |
| Merged robot URDF | `assets/robots/full_robot_gen/<combo dir>/<combo dir>.urdf` (+ `_render.urdf` when the mount has render parts) | `python dexengine/robots/urdf/merge.py <arm>+<hand>` |
| Joint-limit-truncated variant | `assets/robots/full_robot_gen/<combo dir>_truncated/` | `dexengine/robots/urdf/truncate.py` |
| Training USD | `<data root>/data/usd_cache/robots/<paths.combo_usd_relpath>` | URDF → USD → flatten → single-layer export → contact reporters → colour check (below) |
| Reach map (IK pool) | `<pool root>/<arm>_<hand>_{right,left}/` (`--what pool_root`) | `bash scripts/build_reach_map.sh <arm> <hand> [GPU] [WORKERS]` |
| One-hand scene templates | per hand | `python dexengine/placement/uni_layout.py --extract-for <hand>` |

### Retargeting rig

`hand_rig.py` hands the vendor URDF to MuJoCo and saves the compiled MJCF. A hand whose spec names `vendor_mjcf`
(Sharpa Wave) starts from the vendor's own MJCF instead (`dexengine/robots/rig/from_mjcf.py`: the same contract, the
palm site and capsule placements stated in the spec). It then applies the PATH A rig convention:

- a 6-DoF floating base;
- palm and fingertip sites;
- collision geoms;
- actuator order;
- side-prefixed meshes.

MuJoCo merges links joined by fixed joints, so a fingertip site goes on the nearest surviving body, at an offset
derived from the URDF chain. The script refuses to write when that offset disagrees with MuJoCo's own forward
kinematics. `--dry-run` reports without writing.

`retarget_prior.yaml` declares the posture prior and the hyperextension clip of the retargeting IK. The file is
required even when the prior is disabled. Its schema owner is `build_prior_conf` in
`dexengine/retarget/ik/prior.py`, which the hand tracker imports. The IK cut stays at neutral. Only the `_truncated` sim
body may bend further back, by the hand spec's `sim_finger_hyperext_allow_deg` (`{mcp, pip, dip}` in degrees, `null` =
at neutral), applied by `truncate.py` through the cut window's `--sim`; the IK and the placement repairs' finger range
never read it.

The rig's hash is the hand's identity. After a rig change, register the new hash. Every stage-1 output of the old hash
must then be regenerated: training refuses trees whose recorded hash differs.

A hand whose finger has more joints than its tip has coordinates can drift, inside the solve, into a mirrored
solution that reaches the same tip. `overrides: branch_window_deg: {joint: [lo, hi]}` keeps such a finger on its
natural branch: after the solve, `dexengine/retarget/ik/branch.py` re-solves that finger alone to the same tip with
the joint inside the window (marker `[BRANCH]`, report `branch_resolve` in `retarget_diagnostics.json`); every other
column of the output is unchanged. DG-5F-S uses it for the thumb opposition joint `joint_1_2` (|q| <= 90 deg).
The reference finger range (`wrist_shift.finger_ranges`) applies the same window, so the repairs that move fingers
after the IK (wrist ↔ desk, f0, cross repair) never put the finger back and the finger range gate checks it. The sim
body's joint limits do not read the window.

### Merged URDF and USD

`merge.py` composes the arm skeleton, one hand subtree per side and the mount joint, all from the specs. The training
USD is then built by `python scripts/build_robot_usd.py` (plan and runner: `dexengine/robots/usd/build.py`). It runs
these steps with the Isaac Lab python:

1. `dexengine/rl/assets/urdf_to_usd.py`: URDF → USD with Isaac Lab's URDF importer and
   `configs/robots/urdf_converter.yaml`.
2. `dexengine/robots/usd/flatten.py`.
3. `dexengine/robots/usd/mimic.py`: single-layer export.
4. `dexengine/rl/assets/contact_report_bake.py`: contact reporters on the hand and end-effector bodies, and the
   canonical name `<combo dir>.usd`.
5. `dexengine/robots/usd/check_colors.py`, which compares URDF visual colours with USD materials.

```bash
python scripts/build_robot_usd.py --arm rby1 --hand <hand> --dry-run   # plan and input check; starts and writes nothing
python scripts/build_robot_usd.py --arm rby1 --hand <hand>              # build (GPU, Isaac Lab environment)
python scripts/build_robot_usd.py --combos rby1_<hand>_truncated   # a variant folder
```

Without `--combos`, `--arm` or `--hand` it builds every registered combo. The output of a canonical folder is the
training USD `<data root>/data/usd_cache/robots/<paths.combo_usd_relpath>`; the dry run checks that the two agree. It also
checks that every URDF mesh exists and that Isaac's importer reads the same file as the repository's mesh resolver.
The importer resolves meshes relative to the URDF in the code tree, so a hand whose vendor meshes live under
`<data root>/data/_stage/` must be built with the data root equal to the code tree. The build is done when every
count on the outputs is complete (`COMBO_USD_DONE` line; exit code 0). Each finished body gets `bake_record.json`
(`build.BAKE_RECORD`): the converter's `.asset_hash`, the URDF sha256 and the md5 of the USD written. The render body
reads the head camera visual from that URDF only while the record holds (`build.baked_from`); a body whose USD or URDF
changed after its bake, or one baked before the record existed, is re-baked.

Other bodies are built from the training USD:

- The render body (`dexengine/robots/usd/render_body.py`, `hand_look.py`): same physics, repainted.
- The split articulation (`dexengine/robots/usd/split_arm.py`).
- Simulation-only derived bodies that differ only in their colliders.

A collider-derived body records the md5 of its source USD in an identity record (`dexengine/robots/derived_usd.py`).
If it was baked from the render body, it also records the md5 of the training USD. The pre-spawn check refuses it when
a source has changed. The render-body generator itself writes no record.

## Registering a new hand

1. **Write `configs/robots/hands/<hand>.yaml`** with every `HandSpec` field and its `rl_recipe` (start from a copy
   of an existing hand file). No code changes: the loader registers it.
2. **Write the mount** `configs/robots/mounts/<arm>+<hand>.yaml` for each arm (a hand without any mount file stops the
   load). Give its source in `provenance`, and mark it `provisional` until it is measured.
3. **Put the vendor URDFs** where `HandSpec.vendor_urdf` points.
4. **Run the registration chain.**
   - Dry run: `DEXENGINE_ARM=rby1 DEXENGINE_HAND=<hand> python dexengine/data/stages.py --register --arm rby1
     --dry-run`. It checks the declarations, including the vendor URDFs, and prints the steps.
   - The steps are R1 rig (`hand_rig.py`), R4 merged URDF (`merge.py`) and R5 combo USD (`scripts/build_robot_usd.py`).
     Run them without `--dry-run`, or one by one.
5. **Register the rig hash** in `RETARGET_HAND_ASSETS`. Measure it with
   `dexengine.core.layout.asset_hash(<rig dir>)`, and record status `canon` and the hand id.
6. **Build the reach map and the one-hand templates** (table above).
7. **Add the hand to the registry tests** (`CANON_HANDS` in `tests/core/test_registries.py`) and run
   `python -m pytest tests/core/test_registries.py`. The tests check that the rig hash matches, that the combo names
   resolve and that the mounts point at registered arms and hands.
8. **Run stage 1 for each dataset** with the new hand (`python -m dexengine.retarget.reretarget <ds> --hand <hand>`).
   Then run the base-layer steps (`stages.py --status`), stage 2 and stage 3. The base-layer status also lists which
   data-rule thresholds depend on hand geometry.
