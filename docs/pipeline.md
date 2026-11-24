# Pipeline stages

What each stage reads, writes and counts as done. Values (thresholds, grids, judge shapes) have a single owner in
code or `configs/`; this page links to it rather than repeating it.

## A full pass for one hand and one dataset

| # | Stage | Paper | Entry point |
|---|---|---|---|
| 1 | **Retargeting**: raw capture → MANO layer → floating robot hands (MuJoCo IK), then the per-hand base layer | Sec. 3.2, 3.3 | `python -m dexengine.data.read_raw` · `python -m dexengine.retarget.reretarget` · `scripts/stage1_retarget.py` (TACO) |
| 2 | **IK grounding**: placement search on a reachability map, coarse path, arm IK, retiming, gate | Sec. 3.3 | `bash scripts/stage2_ground.sh` |
| 3 | **RL in Isaac Lab**: residual PPO per bundle, cold start; fresh-process judge decides clip acceptance | Sec. 3.4 | `python scripts/stage3_rl.py` · `bash scripts/rl/judge_sidecar.sh` |
| 4 | **Recording**: rollouts of accepted clips, pose and visual augmentation, LeRobot datasets | Sec. 3.5 | `python scripts/stage4_record.py` |

Identity inputs (arm, hand, dataset, embodiment `right` | `left` | `bimanual`) have no defaults. Each entry point takes
the ones it needs as options or from `DEXENGINE_*` variables and stops, naming the input, when one is missing. A
typical pass for one hand and one dataset:

```bash
# 1. raw capture -> MANO layer -> first hand (HOT3D), then the per-hand base-layer steps
python -m dexengine.data.read_raw hot3d read
python -m dexengine.data.read_raw hot3d intake xhand_pro
DEXENGINE_ARM=rby1 DEXENGINE_HAND=xhand_pro python dexengine/data/stages.py --status --datasets hot3d
#    after the base-layer steps (through step 6): the long-clip pieces and aerial tails, cut from their parents. On a
#    fresh data root the axis hand (python -m dexengine.data.rules --axis-hand) runs this first (docs/pipeline.md)
ARM=rby1 CAMPAIGN_DIR=logs/split python -m dexengine.data.stage1.split_apply xhand_pro --datasets hot3d
#    TACO: python scripts/stage1_retarget.py --bases clips.txt --arm rby1 --hand xhand_pro --dataset taco --emb bimanual
#    another hand on an ingested dataset: python -m dexengine.retarget.reretarget hot3d --hand wuji2, then intake
# 2. IK grounding: the reach map and the one-hand templates (once per arm and hand), the candidate clips, then the
#    chain (add --dry-run to print the resolved layers and stop). The tag G1<letters>_<ledger axis>_<digits> lets
#    --fold fold the chunk into the ledgers the RL manifest reads.
bash scripts/build_reach_map.sh rby1 xhand_pro 0
python dexengine/placement/uni_layout.py --extract-for xhand_pro  # right/left embs; reads a bimanual base scene
DEXENGINE_HAND=xhand_pro python dexengine/data/candidates.py --dataset hot3d --emb bimanual --out clips.txt
bash scripts/stage2_ground.sh --arm rby1 --hand xhand_pro --dataset hot3d --emb bimanual --clips clips.txt --gpu 0 \
    --tag G1a_$(python -m dexengine.data.rules --ledger-key hot3d --emb bimanual)_01 --fold \
    $(python -m dexengine.data.rules --launch-flags hot3d)
# 3. RL: train one bundle, then judge it with the same environment (whole hands: dexengine/rl/campaign.py)
python scripts/stage3_rl.py b01 --side both --arm rby1 --hand xhand_pro --dataset hot3d --emb bimanual \
    --seq-dir <trees dir> --seq-list bundle_01.txt --gpu 0
env DEXENGINE_ARM=rby1 DEXENGINE_HAND=xhand_pro DEXENGINE_DATASET=hot3d DEXENGINE_EMBODIMENT=bimanual SIDE=both \
    SEQ_DIR=<trees dir> DEXENGINE_SEQ_LIST=bundle_01.txt RLJUDGE_GPUS="1" bash scripts/rl/judge_sidecar.sh b01
# 4. recording of one finished campaign bundle, then appending it to the hand x dataset LeRobot dataset
python scripts/stage4_record.py bundle --hand xhand_pro --campaign <campaign> --bundle <bundle> --gpu 0 \
    --out <out> --round orig --queue-root <queue root>
python scripts/stage4_record.py merge append --final <data root>/data/hot3d/collect/rby1_xhand_pro \
    --piece <out>/rounds/orig/dataset --piece-id <piece id>
```

## Names used in this repository

| Paper (Sec. 3) | Repository | Entry point | Package |
|---|---|---|---|
| Data preparation (3.2): align, divide, filter | raw → MANO layer, intake, base-layer steps | readers, `python -m dexengine.data.read_raw`, `dexengine/data/stages.py` | `dexengine.data` |
| Hand retargeting and initial pose (3.3) | stage 1: retargeting; f0 repair | `scripts/stage1_retarget.py`, `python -m dexengine.retarget.reretarget` | `dexengine.retarget` |
| Scene placement, arm IK, retiming (3.3) | stage 2: IK grounding (placement judge → bake → retime → gate) | `scripts/stage2_ground.sh` | `dexengine.placement` |
| Physics correction with residual RL (3.4) | stage 3: RL (training, fresh-process judge, campaign) | `scripts/stage3_rl.py`, `scripts/rl/judge_sidecar.sh` | `dexengine.rl` |
| Train-ready data generation (3.5) | stage 4: recording (rollouts, augmentation, render, store) | `scripts/stage4_record.py` | `dexengine.record` |

- **Judge** has two meanings: the *placement judge* (stage 2, `dexengine.placement.judge`) searches placements; the
  *RL judge* (stage 3, `dexengine.rl.judge`) scores checkpoints and decides clip acceptance.
- **Stage** means one of the four entry points. The per-hand steps of stage 1 are numbered 1–9
  (`dexengine.data.stage1`).
- **Clip and tree.** A *clip* is one task segment. A *tree* is a clip's folder after grounding (clip name +
  `dexengine.core.paths.TREE_SFX`). `dexengine.core.paths.ClipPaths` builds every per-clip path from either name.
- **Done** means counts match: outputs = inputs minus what a ledger explains (exclusion or status ledger).
- **Markers.** A switch that changes a run prints a bracketed marker (`[SANDBOX]`, `[COLL_PRESET]`,
  `[HAND_RECIPE]`, …). Identity inputs (arm, hand, dataset, embodiment) have no defaults.

## Data preparation: raw capture → MANO layer

- **Reads.** `data/<ds>/raw` (TACO: `data/taco/raw_unzipped`) and the MANO models in `data/body_models/mano`.
- **Writes.** `data/<ds>/mano/<emb>/<clip>/{task_info.json, 0/trajectory_keypoints.npz}`, the object meshes
  `data/<ds>/assets/objects/<obj>/{visual.obj, convex/}`, and the intake ledger `data/<ds>/INTAKE/`. The `stage` and
  `hoi` drivers produce the object meshes. The `mano` readers write only the clips, and
  `python -m dexengine.data.objects.build` then builds the object meshes their `task_info.json` points at from the raw
  meshes.
- **How.** One reader per dataset, run by one of four drivers (`dexengine.data.readers.READERS`). See
  [`datasets.md`](datasets.md).
- **Intake.** `python -m dexengine.data.intake` checks a delivery: manifest hashes, a registered hand-asset hash,
  `task_info.json`, scene files and trajectory shapes. With `--ingest` it hard-links only the accepted clips into the
  layout.

## Stage 1: retargeting

- **TACO.** `python scripts/stage1_retarget.py --bases <list> --arm <arm> --hand <hand> --dataset taco --emb bimanual`.
  The list is every clip of the TACO source (`python -m dexengine.data.readers.taco_source clips --out <list>`); the
  source, the object meshes and the convex parts are shipped inputs
  ([`datasets.md`](datasets.md#shipped-inputs): download, folders, digests).
  It runs the phases of `dexengine.retarget.batch`: lift, convert, kinematic, decompose, detect-contact, generate-xml,
  ik. Output: `data/taco_retarget/processed/taco/<spec>/bimanual/<clip>/0/trajectory_kinematic.npz`. The run is done
  when `<log>/STATUS` reads `DONE` (every listed clip has its npz). Exit code 0 means done, 1 not done, 2 bad input.
- **Other datasets, first hand.** For the `stage`-driver datasets, `python -m dexengine.data.read_raw <ds> intake
  <hand>` retargets the first hand in its staging folder and ingests it together with the MANO layer.
- **Other datasets, any hand.** `python -m dexengine.retarget.reretarget <ds> [WORKERS] --hand <hand>` needs
  `data/<ds>/mano` and `data/<ds>/assets/objects`. It links them into the hand's stage
  (`--what retarget-stage --hand <hand>`), runs generate-xml and ik there, and writes `processed/MANIFEST.sha256`.
  Every clip without output must be listed in the exclusion ledger
  (`dexengine.core.rules.ledger.retarget_excluded`); otherwise the run exits with code 4. Then admit the result:
  `python -m dexengine.data.intake <stage>/processed --hands <spec> --datasets <ds> --ingest --handoff-id
  reretarget_<hand>`, adding `--supersede-label` when it replaces an earlier generation. It lands in
  `data/<ds>/retarget/<spec>/<emb>/<clip>/` (`task_info.json`, `scene.xml`, `0/trajectory_kinematic.npz`).
  `reretarget <ds> --unstage` then removes the stage links.
- **The same in one step.** `python -m dexengine.retarget.admit <ds> --hand <hand> --retarget [--workers N]
  [--supersede-label <label>]` runs retargeting, the stage manifest, intake, the stage stamps
  (`.stage1_rig_sha256`, `.stage1_prior_sha256`: the rig and prior the stage was admitted with) and the link removal,
  and checks each step's marker. The first step that fails or lacks its marker stops the chain with its exit code
  (8 = retargeting ran another scope, 5 = manifest, stamps or link removal, 9 = no `[INTAKE_SUPERSEDE]` with a
  label, otherwise the step's own code). Logs: `<log dir>/<ds>_1<step>.log`; last line
  `[ADMIT] ds=<ds> hand=<hand> steps <k>/<n> rc <rc>`. Without `--retarget` it admits a stage that is already
  retargeted.
- **Rig identity.** The rig is `assets/robots/hands/<spec>/mjcf`. Its hash must match
  `dexengine.core.layout.RETARGET_HAND_ASSETS`, and trees record the hash so that a later stage can refuse a mix.

### Per-hand base-layer steps

`DEXENGINE_ARM=<arm> DEXENGINE_HAND=<hand> python dexengine/data/stages.py --status --datasets <ds>` prints a table
with one cell per step and dataset, and names the first empty step. `--emit-run <step> --datasets <ds>` prints that
step's argv lists as JSON. The step table is `dexengine.data.stage1.table.STAGES`. The split is not a row of that
table, so `--status` goes from step 6 to step 7 without naming it: run it after step 6. In order:

| Step | Module | Output |
|---|---|---|
| 2 base layer | `dexengine/data/base_adapter.py` | `data/<ds>/base/<spec>/<emb>/<clip>`: yaw alignment, duplicate objects, articulation groups, segment windows |
| 2b, 2c articulated objects | `dexengine/data/oakink2/artic_convert.py`, `dexengine/data/artic/arctic_align.py` | part pairs → one articulated object; joint values |
| 2d contact sidecar | `dexengine/data/contact/sidecar.py` | `precompute/<ds>/contact_mano_all` |
| 2e wrist tilt | `dexengine/placement/repair/wrist_desk_stage.py` | wrist repair where the wrist housing enters the desk |
| 3, 4 cases and roles | `dexengine/data/cases/taxonomy.py`, `dexengine/data/object_roles.py` | active hands, hand–object assignment, object roles |
| 4b quality | `dexengine/data/filtering.py` | quality flags and the training-candidate flag (paper Sec. 3.2 clip filtering) |
| 5 contact sidecar | `dexengine/data/contact/sidecar.py` | `precompute/<ds>/contact_mano_base` |
| 5b support gap | `dexengine/data/support/gap.py` | support gap of raised-support scenes |
| 6 wrist cache | `dexengine/placement/fk/wrist_poses.py` | `precompute/<ds>/robots/<spec>/wrist_poses_<side>/<clip>.npz` |
| split (after step 6, not a table row) | `python -m dexengine.data.stage1.split_apply <hand>` (`ARM`, `CAMPAIGN_DIR`) | long-clip pieces `<parent>_t<n>` and aerial tails, cut from the hand-independent registry `data/<ds>/INTAKE/units/<emb>.json` (measured on the contact timeline); on new pieces it then reruns the all-object contact sidecar, cases, roles, quality (4b), the main-object sidecar, support gap (5b) and wrist cache (6) |
| 7 f0 shadow | `dexengine/retarget/f0/prejudge.py` | `precompute/<ds>/f0_stage/processed/<ds>/<spec>/…`: first-frame penetration repair (Sec. 3.3, initial pose) |
| 8 reach map (per combo) | `bash scripts/build_reach_map.sh <arm> <hand> [GPU] [WORKERS]` | `precompute/taco/ikpool/<arm>_<hand>_{right,left}` (`--what pool_root`) |
| 8b one-hand templates | `python dexengine/placement/uni_layout.py --extract-for <hand>` | idle-hand block for one-hand trees |
| 9 trees | stage 2 below | |

**The split registry.** The registry is written once per dataset and embodiment by one *reference hand*; every other
hand only compares its own cut with it and stops with `✗ differs -- the reference hand runs stage1.split_apply first`
when the registry is missing or differs. On a fresh data root the reference hand is the axis hand
(`python -m dexengine.data.rules --axis-hand`): run its `split_apply` first. To start with another hand, define the
registry with that hand first, for every embodiment `split_apply` visits (`bimanual`, `right`, `left`):
`DEXENGINE_ARM=<arm> DEXENGINE_HAND=<hand> python -m dexengine.data.segment.split --define --datasets <ds> --emb <emb>`;
that hand is then the registry's reference hand.

The f0 shadow is a repaired copy. The base layer is never modified, and stage 2 reads the shadow, so the placement
judge and the bake see the same trajectory.

## Stage 2: IK grounding

```bash
DEXENGINE_HAND=<hand> python dexengine/data/candidates.py --dataset <ds> --emb <emb> --out <list>
bash scripts/stage2_ground.sh --arm <arm> --hand <hand> --dataset <ds> --emb <emb> --clips <list> --gpu <n> \
    --tag G1a_<axis>_01 --fold $(python -m dexengine.data.rules --launch-flags <ds>)   # --dry-run: resolve, print, stop
```

- **Clip list.** `<list>` is the candidate list of `dexengine/data/candidates.py` (the first-pass rules: a known support
  surface and the quality step's training-candidate flag). Stage 2 refuses a clip that is not a candidate unless
  `--allow-noncandidate` is given.
- **Chunk tag.** `--tag` names the chunk: `G1<letters>_<axis>_<digits>`, where `<axis>` is the ledger axis of the
  dataset and embodiment, `python -m dexengine.data.rules --ledger-key <ds> --emb <emb>` (for example `hot3d`,
  `hot3d-right`, `arctic-artic`). Without `--tag` the run gets a `VAL_<time>` tag and stops with exit code 2 before
  judging; `--allow-tag-mismatch` runs it anyway, but its results are never folded into the ledgers.
- **Before the first run** of an arm and hand: the reach map (`bash scripts/build_reach_map.sh <arm> <hand> [GPU]`,
  step 8) and, for the `right` and `left` embodiments, the one-hand templates
  (`python dexengine/placement/uni_layout.py --extract-for <hand>`, step 8b). Without them stage 2 exits with code 2.

- **Reads.** The base layer, the f0 shadow and wrist cache, the reach map, and the search conditions in
  [`configs/placement/judge.yaml`](../configs/placement/judge.yaml). The constraint functions in
  `dexengine/placement/collision/core.py` are shared by the placement judge and the bake.
- **Chain.**
  1. f0.
  2. Placement judge (GPU, Isaac Lab python). It screens placement cells with the reach map (Fig. 3), solves a coarse
     path by dynamic programming over stored IK solutions (Fig. 4a), then confirms with per-frame arm IK from the
     coarse path and random starts.
  3. Bake: placement, arm-IK caches, cross repair.
  4. Retiming to the joint velocity limits (Fig. 4b, `dexengine.placement.retime`).
  5. Gate.
  6. Status ledger.
  7. With `--fold`: the fold of this hand's chunk ledgers of the dataset into its folded ledgers
     (`dexengine/placement/status_fold.py`, marker `[FOLD]`), which the campaign manifest reads. A failed fold ends
     the run with rc 8 and prints the fold command; the chunk's results stand.
- **Writes.**
  - Placements and caches under `precompute/<ds>/robots/<combo>/`: `placement/<tree>.adopted.json`,
    `arm_ik_poses_<side><suffix>/<tree>.npz` (suffix: `DEXENGINE_ARM_IK_DIR_SUFFIX` in `configs/rl/recipe.yaml`) and
    `wrist_poses_<side>/<tree>.npz`.
  - The trees: `data/<ds>/trees/<combo>/<emb>/<tree>/0/trajectory_kinematic.npz` and `task_info.json`.
  - The gate table (`dexengine.core.paths.gate_table`).
  - A chunk folder (run layer `run_chunk`) with `status_ledger.tsv` and `summary.txt`.
- **Gate.** Recomputed per side from the final joint trajectory: the p99 and maximum of the joint-speed-to-limit
  ratio, and the share of grasp frames with a large wrist error. Thresholds: `dexengine/placement/gate/core.py`.
- **Status ledger.** Each input clip gets exactly one terminal state (`dexengine.placement.ledger_states.STATES`):
  `ADOPTED`, `EXHAUSTED` (proof that no placement exists), `UNPROVEN` (budget reached), `GUARD_FAIL`,
  `SOURCE_DEFECT` or `BAKE_ERROR`. Only `ADOPTED` trees go to training.
- **Budget retry.** After the status ledger, stage 2 judges the `UNPROVEN` clips that the cell budget
  stopped (candidates within twice the covered cells, or the search reached confirm) once more with the budget
  doubled (1024 → 2048, `datasets.EXHAUST_RETRY`), continuing from the cell where the search stopped. They run in
  a sibling chunk named by the retry tag (`G1x_<axis>_<n>` → `G1xr_<axis>_<n>`); the fold takes the newer chunk.
  A clip still `UNPROVEN` at 2048 stays `UNPROVEN` (terminal). The round is always on; the marker
  `[RETRY_BUDGET] r1 exhaust_max=2048 n=<targets>` is in `summary.txt` (`n=0` when there is nothing to retry).
  Owner: `dexengine/placement/judge/retry.py`.
- **Re-selection.** A clip whose every bake candidate overlapped the robot spot is judged once more with the robot
  spot disc widened by its recorded margin: `--place-reselect <record>` (run.env `place_reselect=`; form, checks and
  values: `dexengine.placement.bake.place_check`; the campaign lanes derive the record with `--chunk-record`). A clip
  still owed it is not judged or baked without it. The post-bake check and the audit keep margin 0.
- **Done.** Judged = baked = gated counts on `summary.txt`. `--sandbox DIR` writes only under DIR.

## Stage 3: RL

- **Bundles.** `python dexengine/placement/train_set/select.py --status-ledger <chunk>/status_ledger.tsv --n <size>
  --partition-prefix <prefix> --combo <combo> --dataset <ds> --emb <emb>` splits the `ADOPTED` clips into bundles.
  The packing rule groups clips by object and shape family (`dexengine/placement/train_set/bundle_pack.py`).
- **Training.** `python scripts/stage3_rl.py <RUN> --side both|right|left --arm … --hand … --dataset … --emb …
  --seq-dir <trees dir> --seq-list <bundle list> --gpu <n>`. The policy is a residual policy in wrist space with the
  arm reset to the kinematic reference (`dexengine.rl.tasks`, PPO from skrl). The recipe comes from
  [`configs/rl/recipe.yaml`](../configs/rl/recipe.yaml) plus `--set KEY=VALUE`; PPO settings from
  [`configs/rl/skrl_ppo.yaml`](../configs/rl/skrl_ppo.yaml); per-hand settings from
  `dexengine.core.hands.HAND_RL_RECIPES`. Every bundle trains from scratch: a campaign launch first proves that the
  bundle's cell and work area are empty.
- **Modes.** `--mode smoke` boots and checks the environment. `--dry-run` writes the manifest and the trainer argv
  without starting anything. `--print` prints the resolved trainer processes.
- **Writes.** Logs and the command record go to `overfit/logs/refbase/`. Checkpoints go to
  `<work_ckpt>/<RUN>/checkpoints/agent_<iter>.pt` (`dexengine.core.layout.runs.RUN_WORK_AREAS`). Both are under the
  run tree (`paths.run_root_of`: the data root for a release or copied data root, the checkout for the canonical
  one), whatever the cwd (configuration.md, "Data root").
- **Campaign.** `dexengine/rl/campaign.py` plans a whole hand by default and runs the loop with `--launch`. It
  trains the bundles, runs a judge sidecar per run, and finalises each bundle into its cell
  `overfit/hands/<hand>/rl_campaigns/<campaign>/<ds>/<bundle>/`. Its input is one `<hand>/manifest.json` per hand,
  which lists the bundles (read by `dexengine.rl.run_queue`). See [`cluster.md`](cluster.md).
- **Campaign manifest.** `python -m dexengine.rl.manifest --hand <hand> --out <dir>` builds it from the hand's
  folded ledgers (stage 2 `--fold`, or `DEXENGINE_HAND=<hand> python3 dexengine/placement/status_fold.py --datasets
  <ds>` after the chunks): every adopted clip of every axis, minus the dataset exclusions, packed into bundles of at
  most `--n` clips (default 64) by the selector's rule (run-key pools, objects kept together, dups recorded). It
  writes `<dir>/<hand>/manifest.json` and `lists/<bundle>.txt|.tsv`, and refuses a non-empty `lists/`.
  `--datasets` limits the scope and `--exclude-optional` applies an optional exclusion. Before launching, check that
  the manifest still describes the data with `python -m dexengine.rl.manifest --verify <dir>/<hand>/manifest.json`
  (rc 0 = same; it re-hashes the ledgers and re-measures run keys, objects and tree fingerprints).
- **Clip-owner ledger.** The per-hand clip-owner ledger `rl_clip_owner.tsv` names, per clip, its state and the
  checkpoint that owns it. Its schema and reader are in `dexengine.rl.acceptance` (`HEAD`, `read_ledger`). Write it
  from finished campaigns with
  `python -m dexengine.rl.owner_ledger --hand <hand> --campaigns <c1>[,<c2>…] --owner-root <dir>`. Per clip, the
  owner is the kept, judged checkpoint (final judges and special tags) with the highest completion rate over every
  bundle that trained the clip; ties go to the later iteration, then the first bundle id. A clip none of whose bundles
  is judged yet is `pending`. The file lands at `<dir>/docs/experiments/hands/<hand>/tables/rl_clip_owner.tsv`; pass
  the root that `python -m dexengine.core.paths --what owner-root` prints when recording should read it. The writer
  refuses to replace a ledger that holds states or generation columns it does not write (exit 3). Name the campaigns
  whose bundles may own clips: every bundle of them that trained a clip is a candidate (there are no generations
  here; a setup that retires old bundles needs its own writer).

### RL judge and clip acceptance

`bash scripts/rl/judge_sidecar.sh <RUN>`, started with the training's environment, judges a run.

- **Fresh process.** Each judge is a new trainer process that loads one checkpoint
  (`scripts/stage3_rl.py --mode judge --ckpt …`). It runs one warm-up batch and then the judge batches. Finished
  environments are parked at frame 0 between batches (PARK, `dexengine/rl/judge/park.py`). A judge is invalid when a
  PARK gate measures a real object fall.
- **Shape and schedule.** The number of environments, the evaluations per clip and the final judges (fractions of the
  last iteration) are `JUDGE_NE`, `MIN_EVALS_PER_CLIP` and `FINAL_FRACS` in
  [`dexengine/rl/judge/metrics.py`](../dexengine/rl/judge/metrics.py).
- **Termination.** The judge ends a rollout when an object leaves its reference (object-tracking termination, gates
  `POSC`/`ROTC` in the recipe). It never uses contact termination (`dexengine/rl/judge/terms.py`).
- **Clip acceptance** (the paper's 10 % rule). A clip passes a judge when at least a tenth of its evaluations, rounded
  half-up and at least one, reach the clip end (`metrics.threshold`, `metrics.passes`). A run accepts a clip when any
  final judge passes it. `python dexengine/rl/judge/metrics.py rule` prints the rule in force.
- **Run value.** The mean aggregate completion rate of the final judges. *best* is the final judge with the highest
  aggregate; *last* is the last final judge.
- **Writes.** `<RUN>_judge.tsv` (one row per judge), `<RUN>_judge.rc` (`DONE <value> pass10=<k>/<n>`, `FAIL …` or
  `INCOMPLETE …`), per-judge `<RUN>_j<kind><iter>_clips.json`, and the judge record `<RUN>_judge_extras.json`, which
  holds the per-clip [complete, evaluations, pass], best, last and the judge shape.

## Stage 4: recording

`python scripts/stage4_record.py <tool> -h` lists each tool's options. The tools:

- **`drive`.** Takes the clip-owner ledger (`--owner`, `--owner-sha`) and records every clip whose owning
  checkpoint is canonical and whose judged completion rate is at least `--min-rate` (default
  `dexengine.record.bundle.MIN_RATE`). This threshold is separate from the judge's pass rule. Bundles are queued over
  the GPUs (`--gpus`). Without `--spec`, it also needs `--hand`, `--tag`, `--order` and `--round`. It needs
  `DEXENGINE_MAIN_ROOT` and `DEXENGINE_RL_QUEUE_ROOT`.
- **`bundle`.** Records one bundle into `--out`, where `<out>/rounds/<round>/dataset` is the piece. Its input is a
  finished campaign cell (`--hand --campaign --bundle`, which uses the judge record's best checkpoint) or the owner
  ledger (`--owner --clips`). It works in phases:
  1. Parallel rollouts per clip in the judge's fresh-process entry. They write low-dimensional records: state,
     reference, policy action, absolute joint targets, contact forces. Rollouts are rerun with more environments for
     clips without a success.
  2. Selection of one successful episode per trajectory (judge success, smallest end error).
  3. Pose augmentation (`--round aug`, `dexengine.record.augment`): new placements, rolled out without retraining.
     Rule A (`placement/judge/canon_fit.py`) shifts the judge origin so the canonical placement is a window cell.
  4. Rendering under the canonical look and random looks (visual augmentation, `dexengine.record.looks`; assets from
     the `assets` tool).
  5. Storing a LeRobot v2.1 piece (`dexengine.record.store`).
- **`merge`.** `merge append --final <final> --piece <out>/rounds/<round>/dataset --piece-id <id>` appends a checked piece to the
  final dataset of a hand × dataset, `data/<ds>/collect/<combo>`. It is idempotent per piece id. `merge verify
  --final <final>` checks the final dataset.
  Top-up (`record/store/merge_topup.py`): an allow-listed clip already in the final takes only new placement variants.
- **`text`.** Writes the language instructions per dataset to `data/<ds>/collect/text/texts.jsonl` (`--track <name>`;
  it needs an Anthropic API key).
- **`scene-layout`**, **`look-mirror`**, **`assets`.** The scene-layout sidecar of final datasets, the look-D asset
  mirror and the visual-augmentation asset root (`AUGVIS_ASSETS`). `assets` fetches the canonical generation whose
  list is kept in `configs/record/augvis_v2/` (ids and one sha256 per file): a file that differs from its row, a missing
  or unlisted file and a missing list die; `--new-generation` makes a new list (the render refuses it until committed).

Viewer tools (retarget venv; not stage-4 tools, nothing reads their videos): `python -m
dexengine.record.viz.mano_render` draws the source MANO hands and objects of a TACO clip and writes the camera sidecar
`<out>.cam.json`; `python -m dexengine.record.viz.retarget_render --cam_json <sidecar>` draws the kinematic retarget
of the same tree from the same camera, for side-by-side checks.

The final dataset follows LeRobot v2.1:

- `meta/`
- `data/chunk-*/episode_*.parquet`: joint state, action, joint targets, object poses, reference, tracking error and
  `video_frame_index`.
- `videos/…/observation.images.{ego_left,ego_right,exo}`.
- Two folders outside the standard: `depth/` (ego depth) and `forces/` (contact forces).
- `head_view/`.

The feature schema is one per hand × dataset (`dexengine/record/store/form.py`). Low-dimensional rows and video frames
have different rates. Read the videos with a LeRobot timestamp tolerance, as described in
`dexengine/record/store/lerobot.py`.

## Data made by older code

Stored values (ledger reasons, verdicts, marks) are English and carry their markers. Data written by older code is
not read as is; the release ships no such data and no migration tool.
