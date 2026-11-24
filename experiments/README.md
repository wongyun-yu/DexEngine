# Paper experiments

Code for the experiments of Sec. 4 of the paper and its supplementary sections. It is not part of the released
pipeline: `dexengine/` is the pipeline (human capture → robot data); `experiments/` produced the paper's numbers on top
of it. The rule is that `dexengine/` does not import `experiments/`: experiment variants that change the simulated body
plug in through `dexengine.rl.tasks.hooks` (RL switch `plugins`: `scripts/stage3_rl.py … --set
plugins=experiments.ablations.floating_hand.plugin`, or `DEXENGINE_PLUGINS=…` in a recorded launch environment,
for the floating hand; `experiments.scoring.objtrace_plugin` for the judge's object-pose trace);
`tests/core/test_no_experiments_import.py` checks the rule.

## Names used in the code

The code keeps the short names the experiments were run under (environment variables, log markers and record keys
carry them, so they are not renamed):

| name | meaning |
|---|---|
| E1–E3, `e123` | Tab. 2 (pipelines on the same 64 TACO clips) and the RL ablations on those clips (Tab. 3, Tab. S8, Sec. E.4–E.7) |
| E5, `e5` | Tab. 4 / Tab. 5: policy training data, fine-tuning and the closed-loop simulation evaluation |
| B0, B0r | the random 64-clip candidate set (`scoring/clips/select_B0_pool.py`) and the same set after the CHORD-schema replacements (`scoring/clips/provenance/`) |
| V1 | the earlier SharpA TACO training run whose bundles and checkpoints some Tab. 2 rows start from (roster `v1_o64`) |
| NA, `na` | floating-hand ("no arm") runs |
| K | clips per RL bundle (K = 64 … 1) |

## Which tree produced the numbers

The scripts here target the **current tree**. The published numbers were produced on earlier commits of the
development history. That history is not part of this public repository: the commits below are recorded for
provenance and cannot be checked out here. Those commits used an older package layout and older environment-flag names
(`experiments/workspace.py` names both). A run on the current tree is a new run: simulator rollouts are not
bit-deterministic, so compare against the same checkpoint (same-checkpoint A/B) before calling a table reproduced.

| commit (development history) | used for |
|---|---|
| `c0c67024a` | Tab. 2 Ours K-sweep training (K = 64 … 1), the training code of every arm-mounted Ours row |
| `8a224f492` | Ours K = 1 at 200 iterations (a row of the Tab. 2 results file, not of the paper table) |
| `6b72aabea` | Tab. 2 Ours judges: `c0c67024a` + the object-pose recorder (`scoring/objtrace_recorder.py`) |
| `e0578ffd9` | floating-hand judges (floating-hand physics + recorder) |
| `a2f900740` | Tab. 2 Ours rows rerun to the clip end with the object-tracking termination off |
| `a022e2adf` | Tab. 3 / Tab. S8 RL ablation knobs on the K = 64 tree (all default off) |
| `8ffdd0d72`, `9f85433d7` | on `a022e2adf`: Tab. 3 "arm re-solved by IK": reset from a per-frame independent IK cache, and its bake |
| `6c31bda4c` | Sec. E.7 sequential-IK arm caches and the arm-IK limits report |
| `9971a549b` | Tab. 3 arm-joint residual rows |
| `b91a13407`, `97246fb96` | Tab. 2 floating-hand Ours row (anchor rule on): trained at `b91a13407`, judged at `97246fb96` with the object-tracking termination off (stored `results/ours_floating/*/meta.json`); `97246fb96` is also the tree of the Tab. S8 floating arms |
| `ef8b74a57` | Tab. S8 floating-hand ablation arms (CONTACT_MATCH / FT_HINGE_CM knobs on the floating tree) |
| `c5dbc2ed3`, `2a60cf9df` | Sec. E.5 placement-search ablation; Sec. E.4 post-hoc arm solve of the floating-hand and SPIDER wrist paths |

Data generations used by these commits: the V1 roster `v1_o64` (34 bundles, 2,120 clips; `NAMED_ROSTERS` in
`dexengine/core/layout/generations.py`) and its precompute generation `sharpa_wave_pre1005`
(`precompute/taco/_generations/sharpa_wave_pre1005`). The frozen clip sets are in `scoring/clips/`
(`E123_CLIPS_v1.tsv` = the 64 Tab. 2 clips, `E123_HELDOUT_v1.tsv` = the 64 held-out clips). Baseline code pins
(SPIDER, EgoEngine, FlashCHORD, ManipTrans, DexMachina, FlashDexRetarget): `baselines/TREES.txt`.

## External workspace

Baseline checkouts, their inputs, run logs and per-row results are not in git. They live in one workspace folder named by
the environment variable `DEXENGINE_E123_ROOT` (owner: `experiments/workspace.py`; no default):

```
$DEXENGINE_E123_ROOT/
  code/<method>/   baseline checkouts at the commits in baselines/TREES.txt (+ their venvs)
  data/<method>/   baseline inputs, converted clips, assets, checkpoints, staged judge inputs
  logs/            run logs
  results/<row>/   per-row records and score files (the table builders read these)
  run/             generated per-clip command files
```

Commands below run from the repository root with `PYTHONPATH=.` (or an installed checkout). The cluster launchers,
job and queue builders and the table builders that read the original run records are site-specific (machine names,
cluster queues, staging paths) and are not included; each row below names the single-machine entry point that is.

## Paper item → code

### Tab. 2 — pipelines on the same 64 TACO clips (SharpA)

| paper item | entry point | command | inputs | results |
|---|---|---|---|---|
| clip set | `scoring/clips/select_B0_pool.py`, `select_heldout_v1.py` | `python -m experiments.scoring.clips.select_B0_pool --owner-tsv <rl_clip_owner.tsv> --out-dir <dir>` | CHORD TACO release list (`scoring/clips/chord_hf_taco_dirs.txt`), our SharpA per-clip table of the canonical TACO clips | `scoring/clips/E123_CLIPS_v1.tsv` (frozen; `scoring/clips/provenance/` has the replacements) |
| ADD-AUC · SP-SR · MP-SR of one run | `scoring/score.py` | `python -m experiments.scoring.score spider\|egoengine\|chord\|motion\|npz …` (FlashCHORD venv: it calls `flash_chord.evaluation.metrics`) | one rollout + its reference | score JSON next to the run (`results/<method>/…`) |
| SPIDER, EgoEngine, CHORD rows | `baselines/common/gen_cmds.sh` | `bash experiments/baselines/common/gen_cmds.sh experiments/scoring/clips/cand_B0r_random_chordfull.tsv $DEXENGINE_E123_ROOT/run` | `data/spider`, `data/egoengine`, CHORD processed motions | `results/{spider,egoengine,chord}/` |
| start state (a) and 0.2 kg inputs | `baselines/common/prep_start_a.py`, `prep_mass_0p2.py` | `python -m experiments.baselines.common.prep_start_a <clips> data/spider data/spider_a`; `<venv python> -I experiments/baselines/common/prep_mass_0p2.py data/spider_a data/spider_a_m0p2` | SPIDER-form clip folders | `data/spider_a*`, `data/egoengine_a*` |
| per-method queues | `baselines/common/mkcmds.py`, `rowlib.sh`, `capbin/` | `python -m experiments.baselines.common.mkcmds --queue <name> <out>` (`chord`, `maniptrans_m0p2`, `maniptrans_imitator`, `maniptrans_retarget_pool`, `spider_a_nocheck_m0p2`, `egoengine_a_m0p2[_r1\|_r2]`); a queue row is one shell command: run it with `run_row` of `baselines/common/rowlib.sh`, which puts the `timeout` / `budget` shims of `baselines/common/capbin/` on its PATH | `run/*_cmds.txt` | queue TSV |
| ManipTrans row | `baselines/maniptrans/gen_cmds.sh` | `bash experiments/baselines/maniptrans/gen_cmds.sh 8192` | imitator pool (`baselines/maniptrans/pretrain_pool.txt`), calibration (`baselines/maniptrans/calib_from_chord.py`) | `results/maniptrans/` |
| DexMachina row | `baselines/dexmachina/` | per clip: `DM_VARIANT=m0p2 bash experiments/baselines/dexmachina/prep_clips.sh …` → the authors' `retargeting/parallel_retarget.py --control_steps 2000 --save_name para` and `retargeting/map_contacts.py` → `bash experiments/baselines/dexmachina/train_clip.sh <key> <T60> 4096 <gpu>` (the authors' recipe; key = `python3 experiments/baselines/dexmachina/variant.py --key m0p2 <idx>`) → their evaluation → `export_rollout.py` → `scoring/score.py`. Table row = variant `m0p2match` (`variant.py`): the port's switch `DEXM_MATCH=1 DEXM_FRIC_R=2.0 DEXM_FRIC_L=1.25 DEXM_MASS_KG=0.2 DEXM_CAP_S=<cap s>`, RL capped at 2 h of training on an NVIDIA L40S (scaled by card speed on another card), score = the END checkpoint, 64 rollouts. The switch lives in port commits that are not exported (`baselines/TREES.txt`), so this row cannot be rerun from the shipped patches alone; the row runner that applies the cap and the evaluation is not included | DexMachina checkout + `baselines/dexmachina/patches/` | `results/dexmachina/records_m0p2match/` |
| FlashDexRetarget row | `baselines/fdr/` | `setup.sh` → `retarget.sh 4` (import) → the authors' TACO pipeline (`$F/scripts/retarget/taco.sh`, F = the FlashDexRetarget checkout) with `filters.wrist_acc_spike.max_acc=200` → the authors' training launcher (`$F/scripts/train_flashdexretarget.sh`) with the overrides of `variants/disks_mass_0p2.txt` (`bash experiments/baselines/fdr/variant.sh overrides <file>`) → `eval_record.py --variant <file>` (64 rollouts per clip) → `score.py`; the training and scoring row runners are not included | FDR checkout | `results/fdr/` |
| SPIDER / EgoEngine physics-matched rows | pinned `code/spider/match_phys/run_clips.sh` (see `baselines/TREES.txt`) | — | `data/spider_a_m0p2` | `results/sim_m0p2/` |
| Ours rows, arm-mounted (K = 64 … 1) | `scripts/stage3_rl.py` (train, then `--mode judge`) on the current tree; the paper runs used the launcher of commit `c0c67024a` | bundles of the frozen 64 clips: K = 64 is one bundle, K = 32 two; each smaller K splits every bundle of the next larger K in half (clips sorted by object key, then name); K = 1 = one clip per bundle | frozen 64 clips | checkpoints and judge records |
| Ours rows from the judge's object poses | `scoring/objtrace_recorder.py` (in the judge, `DEXENGINE_E123_OBJTRACE=<npz>` with `DEXENGINE_PLUGINS=experiments.scoring.objtrace_plugin`), `scoring/ours_objtrace.py` | `python -m experiments.scoring.ours_objtrace <objtrace.npz> <tree root> <object assets> <out> [64]` | judge object-pose file | `results/ours_rl/…/<clip>.json` |
| to-end rerun clip lists | `scoring/k_sweep/rerun_clips.py` | `python -m experiments.scoring.k_sweep.rerun_clips` | `results/ours_rl`, `results/ours_v1prefill` | the clips judged again (`results/ours_rl_off/`) |
| GPU-minutes per clip (L40S) | `scoring/cost/dexmachina_onetime.py`, `na_time.py`, `dexmachina_fps_row.sh` | — | the runs' own logs | per-run times; the card-speed ratio table was built from the original run records (not included) |
| **the table** | built from the original run records (not included) | — | record lines + workspace `results/` | `results.json`, `results_mass0p2.json`, `results_match.json` |

### Tab. 3 — arm attachment in RL; Tab. S8 / Sec. E.6 — other recipe changes

| paper item | entry point | command | inputs | results |
|---|---|---|---|---|
| ablation arms (one row each) | the RL launcher of commit `a022e2adf` (or of the floating tree, Tab. S8) with one knob switched on per arm (all knobs default off there); the arm list and the host and cluster runners are not included | — | the K = 64 bundle | judged checkpoints at 400 · 800 · 1200 · 1600 iterations |
| "arm re-solved by IK at random starts" cache | `ablations/rl_arm/arm_ik_cold_frames.py` | `python -m experiments.ablations.rl_arm.arm_ik_cold_frames …` (see `--help`) | canonical arm-IK caches | `arm_ik_poses_<side>_a2cold/` + `FLIPS.json` (per side, the share of adjacent frames whose largest arm-joint jump exceeds 10 / 20 / 45°) |
| judge object poses of one arm | `scoring/objtrace_recorder.py`, `scoring/ours_objtrace.py` (as for Tab. 2) | — | finished training run | object-pose files, scored into `results/ours_rl_abl/` |
| **the table** | built from the original run records (not included) | — | `results/ours_rl_abl/<job>/<alias>/` | ablation table |
| held-out transfer probe | `ablations/rl_arm/heldout/ho_table.py` | `python -m experiments.ablations.rl_arm.heldout.ho_table` | `results/ours_rl_ho/` (clips `scoring/clips/E123_HELDOUT_v1.tsv`) | `heldout_table.{json,md}` |

### Sec. E.4 — floating hand + post-hoc arm IK ("26 % fewer clips"); Tab. 2 floating-hand row

| paper item | entry point | command | inputs | results |
|---|---|---|---|---|
| floating-hand body (six virtual wrist joints) | `ablations/floating_hand/` (`plugin.py` registers it with `dexengine.rl.tasks.hooks`), `make_floating_urdf.py`, `build_floating_usd.sh` | `python -m experiments.ablations.floating_hand.make_floating_urdf --hand sharpa_wave`; `HAND=sharpa_wave bash experiments/ablations/floating_hand/build_floating_usd.sh`; runs set `DEXENGINE_ARM=floating` and the switch `plugins=experiments.ablations.floating_hand.plugin` | the combo URDFs | floating URDF / USD assets |
| Tab. 2 "DexEngine (Ours)" floating row (K = 64, 2.8 GPU-min) | `scripts/stage3_rl.py` with `DEXENGINE_ARM=floating` and `--set plugins=experiments.ablations.floating_hand.plugin`, timed by `scoring/cost/na_time.py` | — | the frozen 64 clips (`scoring/clips/E123_CLIPS_v1.tsv`) | checkpoints, judge trajectory record, object-pose file → `scoring/ours_objtrace.py` |
| post-hoc reference set | `ablations/posthoc/posthoc_refs.py`, `spider_f0_wrists.py`, `spider_f0_wrist_caches.py` | `python -m experiments.ablations.posthoc.posthoc_refs …` | trajectory record of the floating-hand judge (or SPIDER rollouts) | sandbox trees + post-hoc arm-IK caches |
| outcome counts | `ablations/posthoc/posthoc_counts.py`, `cfull_replay_agg.py` | `python -m experiments.ablations.posthoc.posthoc_counts …` | `scripts/stage2_ground.sh --judge-only --judge-extra "--place-strategy-module experiments.ablations.placement.place_strategy --place-strategy adopted …"` run on the sandbox | `cfull_posthoc_counts.json`, `b3_spider_posthoc_counts.json` |

### Sec. E.5 — placement search; Sec. E.7 — sequential IK

| paper item | entry point | command | inputs | results |
|---|---|---|---|---|
| search strategies (rows a–e) | `ablations/placement/place_strategy.py` (behind `dexengine` placement judge `--place-strategy`), `rows.py` | — | — | — |
| sample of 128 clips | `ablations/placement/draw.py`, `stage_list.py` | `python -m experiments.ablations.placement.draw …` | folded status ledgers | `ablations/placement/data/lists/clips_main.tsv` |
| runs | per item, the full chain f0 seed → judge → bake → retime → gate: `scripts/stage2_ground.sh --arm rby1 --hand wuji2 --dataset <ds> --emb <emb> --clips <list> --out <dir> --sandbox <dir> --judge-extra "--place-strategy-module experiments.ablations.placement.place_strategy <row flags of rows.py>"`; the multi-host queue runner is not included | — | clip lists | per-item records |
| **the table** ("43.8 % vs 1.6 %", "45.3 %, 3× longer") | `ablations/placement/aggregate.py`, `aggregate_rows.py`, `sample128_vs_all.py` | `python -m experiments.ablations.placement.aggregate --fetched <records> --lists <dir> --root <data checkout> --json <table json>` | per-item records, production ledgers | table JSON |
| placement funnel by hand | `ablations/placement/drop_stages.py` (+ `scoring/judge_fail_detail.py`) | `python -m experiments.ablations.placement.drop_stages --root <data> --raw <dir> --out <dir>` | folded ledgers, judge failure classes | `b1_drop_stages.{json,tsv}` |
| sequential-IK caches ("28 % vs 89 % within limits") | `ablations/rl_arm/arm_ik_seq_frames.py`, `arm_ik_limits_report.py` | `python -m experiments.ablations.rl_arm.arm_ik_limits_report --arm rby1 --hand sharpa_wave --ik-right <cache> --ik-left <cache> --trees <trees> --out <json>` per cache | canonical + sequential arm-IK caches (built at `6c31bda4c`) | one limits report per cache |
| sequential-IK RL rows | ablation arms as in Tab. 3 (s0 · s1 · s1r) | — | sequential arm-IK caches | judged checkpoints |

### Tab. 4 / Tab. 5 / Fig. 6 — policies (`policies/`, `sim_eval/`, `real_kit/`)

| paper item | entry point | inputs | results |
|---|---|---|---|
| Tab. 4 "Kinematic replay data" | `policies/kin_data/` (prep → collect → record) | our clips and placements, kinematic reference | LeRobot datasets |
| Tab. 4 "SPIDER + arm data" | `policies/spider_arm/` (`prep_inputs.py`, `data.py`) | SPIDER runs on the 60 task clips | LeRobot datasets |
| Tab. 4 "DexEngine data" | `policies/data/build_formal.py`, `convert.py`, `manifest*.py`; dense L1 render `policies/dense_render.py` | recorded pipeline data | LeRobot datasets |
| DP / GR00T fine-tuning | `policies/dp/train.py`, `policies/gr00t/train.py` (recipe A), `policies/mid/` (mid-training corpus) | datasets | checkpoints |
| Tab. 4 closed-loop success | `sim_eval/eval_run.py` (+ `sim_eval/batch.py`, `run_scene.py`, `judges/`), summary `sim_eval/report/dpf_summary.py` | checkpoint, L0/L1 scene lists (`sim_eval/lists/`) | `<run>/<LEVEL>.json`, report JSON |
| Tab. 5 real fine-tuning data | `real_kit/v4/`, `real_kit/vis_*.py` | recorded data | real-robot fine-tuning sets (deployment code is outside the repository) |

## Provenance tools (run by hand, nothing imports them)

These one-shot scripts produced inputs or checks that the paper's records cite. No pipeline step or other module runs
them; they stay so the cited outputs can be regenerated.

| script | what it produced |
|---|---|
| `policies/kin_data/reach_table.py` | E5 kinematic baseline IK reach table (dataset × hand), cited by the E5 formal-set results |
| `policies/mid/convert_corpus.py` | GR00T mid-training corpus conversion (sample and full corpus) |
| `policies/data/kin_manifest.py` | kinematic conversion list of one E5 formal task (`build_formal.py --kin-manifest` reads its output) |
| `sim_eval/scenes/scenes_count.py` | E5 scene-set check (set differences) and scene list writer |
| `sim_eval/scenes/variant_src.py` | inputs of the E5 placement-variant scenes (`prep.py --variant-src` reads its output) |
| `baselines/maniptrans/hull_cache.py` | the authors' object convex hulls for the ManipTrans rows (`data/maniptrans/hull_cache`) |

## Layout

```
experiments/
  workspace.py   the external workspace (DEXENGINE_E123_ROOT)
  scoring/       Tab. 2: shared scorer, Ours object-pose scoring, GPU-minutes cost, to-end reruns, clip sets
  baselines/     Tab. 2 prior methods: runners, adapters, patches, upstream pins (TREES.txt)
  ablations/     Tab. 3 / S8 RL ablations (rl_arm), floating-hand variant (floating_hand),
                 Sec. E.4 post-hoc arm IK (posthoc), Sec. E.5 / E.7 placement search (placement)
  policies/      Tab. 4 / 5 training data and policy training
  sim_eval/      Tab. 4 closed-loop evaluation
  real_kit/      Tab. 5 real-robot fine-tuning data
```
