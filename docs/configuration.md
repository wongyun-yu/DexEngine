# Configuration

Configuration comes from four places:

- typed YAML sections;
- a few stage-specific files;
- command-line options;
- a short list of environment variables that must cross a process boundary.

Values live in exactly one of these. A key that no consumer reads, a value of the wrong type, and an identity input
without a value all stop the run instead of being ignored.

## Typed sections: `dexengine.core.config`

A section is a dataclass whose field defaults are the values used when nothing is configured.

```python
from dexengine.core import config
cfg = config.load(SlurmConfig, "cluster/slurm")                       # configs/cluster/slurm.yaml, if present
cfg = config.load(SlurmConfig, "cluster/slurm", overrides=["partition=gpu", "extra_sbatch=[--exclude=<node>]"])
```

Priority, lowest first:

1. the dataclass defaults;
2. `configs/<name>.yaml`, or an explicit file. A missing default file is not an error; a missing explicit file is;
3. `KEY=VALUE` overrides. A dotted key reaches a nested dataclass. The value is parsed as YAML, so `true`, `3`,
   `[a, b]` and `{k: v}` are typed.

Unknown keys and wrong types raise `ConfigError`. On a command line, `config.add_cli(parser)` adds `--config FILE` and
`--set KEY=VALUE` (repeatable), and `config.from_args(cls, name, args)` loads with them. `config.dump` and
`config.to_yaml` give the effective configuration, for recording it next to an output.

| Section | File | Loaded by | CLI |
|---|---|---|---|
| `LayoutConfig` | `configs/layout.yaml`, or the file named by `DEXENGINE_LAYOUT_CONFIG` | `dexengine.core.layout.roots` | (none; see the data root below) |
| `SlurmConfig` | `configs/cluster/slurm.yaml` | `python -m dexengine.cluster slurm` | `--config`, `--set` |
| `K8sConfig` | `configs/cluster/k8s.yaml` | `python -m dexengine.cluster k8s` | `--config`, `--set` |

### Scene desk

Every scene uses one desk: the Willow table of NVIDIA's Commercial Assets Pack. It is visual only; the desk collides
through five cuboids whose sizes live in `dexengine.core.desk`. The repository does not carry the table (NVIDIA terms:
no redistribution). Download it once per data root, together with the look D materials:

```bash
python -m dexengine.record.looks.look_d --mirror   # each user downloads it from NVIDIA under NVIDIA's terms
```

The files land in the look mirror under the data root (`dexengine.record.looks.look_assets.MIRROR_REL`). The code
pins their sha256 (`look_assets.WILLOW_SHA256`). A missing or changed file stops the run with a `ConfigError` that
names the download step. Every Isaac Lab process prints a `[DESK]` line with the file it spawned. There is no desk
choice: `--set scene.…`, `DEXENGINE_DESK` and `DEXENGINE_DESK_USD` are refused.

## Stage-specific configuration

| File | Read by | How to override |
|---|---|---|
| `configs/placement/judge.yaml` | placement judge (`dexengine.placement.judge.config`): grid, screening, coarse path, confirmation, exhaustive search, desk | each value is the default of the judge flag of the same name; another file with the judge's `--config <yaml>`; stage 2 passes extra judge flags with `--judge-extra "…"` |
| `configs/placement/canon_env.sh` | the placement drivers' shell environment | (sourced; clears experiment overrides) |
| `configs/rl/recipe.yaml` | RL launcher `dexengine.rl.launch.recipe`: tracking gates, action and plant, checkpoint interval, collision preset, … | `scripts/stage3_rl.py … --set KEY=VALUE` (repeatable) |
| `configs/rl/skrl_ppo.yaml` | PPO agent (skrl) | (edit the file) |
| `configs/pipeline.env` | shell stages (`scripts/_lib.sh`) | the environment, then options (`--arm`, `--hand`, `--dataset`, `--emb`, `--gpu`, `--seq-dir`, `--main-root`); another file with `PIPELINE_CONFIG=<file>` |
| `configs/data/read_hoi.yaml` | `python -m dexengine.data.read_raw.hoi` | `--config` |
| `configs/datasets/…`, `configs/contact/…` | per-dataset tables and measured contact operating points, read by their modules | (data, not knobs) |
| `configs/record/text/` | language-annotation prompt and per-dataset vocabulary (`dexengine.record.text`) | (edit the files; per-track settings live in the track's `track.json`) |
| `configs/robots/urdf_converter.yaml` | URDF → USD conversion | (fixed; regenerate every USD after a change) |

**RL recipe.** `configs/rl/recipe.yaml` has a `common` column and per-path columns: `uni` for one hand, `bi` for
both hands. It also has a `launcher` table of launcher-only defaults. Keys that start with `DEXENGINE_` are handed to
the Isaac Lab trainer process as environment variables. Every other key becomes part of the trainer's argv. Values a
caller gives with `--set` win when they are non-empty. Per-hand settings come from
`dexengine.core.hands.HAND_RL_RECIPES`, and the launch prints them as `[HAND_RECIPE]`.

**Shell stages.** `configs/pipeline.env` accepts only the keys in `PIPELINE_CONFIG_KEYS` (`scripts/_lib.sh`). Any
other key, or a line that is not `KEY=VALUE`, stops the stage. Empty values mean "required": the stage dies and names
the option to give.

## Data root

All data (`data/`, `precompute/`, `overfit/`, `logs/`) lives under one data root, resolved in this order:

1. `DEXENGINE_MAIN_ROOT`;
2. `data_root` in `configs/layout.yaml`;
3. the code checkout.

A code-only checkout (a second clone or a worktree) names the data root and uses no symlinks.

Run outputs — the RL work areas (`overfit/logs/refbase` launch logs and manifests, `logs/skrl/dexengine` run folders),
the hand roots under `overfit/hands/`, default video folders (`results/taco/videos`) and relative output options
(`--videos_out_dir`, `--log-dir` of `dexengine.retarget.admit`) — go to the run tree, never the cwd.
The run tree follows the kind of the data root (`dexengine.core.paths.run_root_of`): the code checkout when the data
root is that checkout or another main checkout (a worktree naming the main checkout keeps its runs apart), the data
root itself for a release or copied data root and for a registered generation, and its base's run tree for a shadow
root. Shells read it with `python -m dexengine.core.paths --what work-area --area work_log|work_ckpt`.

Every stage entry point takes `--data-root DIR`:

- `scripts/stage1_retarget.py`, `scripts/stage3_rl.py`, `scripts/stage4_record.py <tool>` (before or after the tool's
  own options), `scripts/build_robot_usd.py`;
- the shell stages (`scripts/stage2_ground.sh`; `--main-root DIR` is the same option);
- the modules they call or the docs name: `python -m dexengine.retarget.run`, `…retarget.reretarget`,
  `…data.intake`, `…data.read_raw`, `dexengine/data/stages.py` and the stage-4 tools (`dexengine.record.driver`,
  `…bundle`, `…store.merge` (before the sub-command), `…store.scene_layout`, `…store.scene_manifest`,
  `…looks.look_d`, `…looks.augvis.assets`, `…text.pipeline`).

They declare it with `config.add_data_root_arg` and apply it with `config.apply_data_root` right after parsing, which
exports it as `DEXENGINE_MAIN_ROOT` for the process and its children. A dispatcher that hands its command line on
uses `config.take_data_root`. No entry module reads the data root at import time, so the value reaches every reader
(`tests/core/test_data_root_cli.py`).

Other commands the docs name take no `--data-root` and read `DEXENGINE_MAIN_ROOT` (else `configs/layout.yaml:
data_root`, else the checkout), so export it when the data root is not the checkout:
`python -m dexengine.data.stage1.split_apply`, `dexengine/data/segment/split.py`, `dexengine/data/candidates.py`,
`python -m dexengine.data.rules`, `python -m dexengine.data.objects.build`, `python -m dexengine.data.readers.<ds>`
(these two also take explicit `--raw`/`--out` paths), `python -m dexengine.data.read_raw.hoi` (it works in its
`--dataset-root` stage), `dexengine/placement/uni_layout.py`, `dexengine/placement/status_fold.py`,
`dexengine/placement/train_set/select.py`, `bash scripts/build_reach_map.sh` and `bash scripts/rl/judge_sidecar.sh`.
`python -m dexengine.release.fromraw` makes its own root (`--root`).

`dexengine.core.layout.roots.check_data_root` refuses an unregistered root that links into another root's data.

**Site layout file.** A site states its own values in a layout file of the same form and exports
`DEXENGINE_LAYOUT_CONFIG=<file>`; it replaces `configs/layout.yaml` for the process and its children. Besides the
data root it holds the record storage roots (`collect_*`), the root of the RL clip-owner ledgers (`owner_root`; empty
= the data root) and the RL campaign hosts (`hosts`, `host_gpu_count_cmd`; empty = this machine only). The release
values are the ones in `configs/layout.yaml`. A process that reads the layout prints the file it read on stderr,
`[LAYOUT] <file> pid=<pid>`, so a site can prove from a log that its processes read its file. Stage 2
with `--sandbox DIR` resolves only the layers it writes under DIR.

## Workers and threads

Every process or thread pool sizes itself in `dexengine.core.capacity`, so a small machine (a 4 GiB pod, a laptop)
runs the same steps as a large one, only with fewer workers. The machine is read when a pool starts: the usable CPUs
are the affinity capped by the cgroup `cpu.max`, and the memory is the smallest cgroup `memory.max`, or the host's
MemAvailable without a limit. Each step has a memory statement (the parent's peak, one worker's peak and the old fixed
count as a cap) in `capacity.NEEDS`; measured statements name their records in `dexengine/core/capacity_peaks.tsv`.
Workers = the smallest of the CPUs, (memory x 0.85 - parent) / per worker, the work items and the cap. An explicit
`--workers` / `--jobs` replaces that value. Process-pool workers run one thread each (`OMP_NUM_THREADS`, the BLAS
variables and `PXR_WORK_THREAD_LIMIT` are set while the pool is open unless you set them; your values are back after
it). A thread pool changes no thread setting. Outputs do not depend on the worker count. Each pool prints on stderr

    [CPU] step=<module> usable=<n>(affinity|cpu.max|share) mem=<GiB>(memory.max|host) parent=… per_job=… cap=…
          workers=<n> threads=<n>(owner|env) or threads=-(process-wide) source=owner|option

when it starts and the same line with `peak_parent=… peak_job=…` when it ends. Memory sources are the host's
MemAvailable, the cgroup (`memory.max`, v1 `memory.limit_in_bytes`), a Slurm job's request (`SLURM_MEM_PER_NODE`, or
`SLURM_MEM_PER_CPU` x `SLURM_CPUS_PER_TASK`) and `DEXENGINE_MEM_GIB`, the smallest wins. When the limit cannot be read
(a container without the cgroup filesystem, or a Slurm job whose site keeps `--mem` out of `memory.max`) a process
pool stops with `[CPU] ✗` instead of planning on the whole host: pass the Slurm variables into the container
(singularity `--cleanenv` drops them: `SINGULARITYENV_SLURM_JOB_ID`, `SINGULARITYENV_SLURM_MEM_PER_NODE`) or set
`DEXENGINE_MEM_GIB`. `python -m dexengine.core.capacity
cpus` prints the usable CPUs for shell scripts.

## Environment variables

**Process-boundary variables.** These are declared in `dexengine.core.config.ENV_VARS` and read with
`config.env(name)`. They exist because a child process inherits them: an Isaac Lab worker builds its scene inside the
gym entry point, at import time, where no command line reaches.

| Variable | Meaning |
|---|---|
| `DEXENGINE_MAIN_ROOT` | the data root |
| `DEXENGINE_ARM`, `DEXENGINE_HAND` | arm and hand of the running process (no default) |
| `DEXENGINE_DATASET`, `DEXENGINE_EMBODIMENT` | data axes of the running process (no default) |

**Variables a user sets.**

| Variable | Needed by | Meaning |
|---|---|---|
| `RETARGET_PY` | stages 1–2, recording | retargeting interpreter (default `<code checkout>/.venv/bin/python`; the data root is not searched) |
| `ISAACLAB_DIR` | Isaac-side scripts | Isaac Lab checkout: `$ISAACLAB_DIR/isaaclab.sh` and an executable `$ISAACLAB_DIR/env_isaaclab/bin/python`. Unset = a sibling `../IsaacLab` of the code checkout (or of the working directory, the data root, the main checkout of a worktree; `scripts/_isaac_py.sh`). Required for a tree installed anywhere else |
| `DEXENGINE_RL_QUEUE_ROOT` | checkpoint loading, `stage4_record.py drive` | RL queue root that declares which tree generation each checkpoint was trained on (`dexengine.rl.checkpoints`) |
| `RLJUDGE_GPUS` | `scripts/rl/judge_sidecar.sh` | GPUs the judge may use (space separated) |
| `RLJUDGE_MIN_AVAIL_GB`, `RLJUDGE_RSS_GB`, `RLJUDGE_MAX_CONC` | judge sidecar host gate | host memory gate and judge concurrency (defaults in `scripts/rl/host_gate.sh`; lower them on machines with less RAM) |
| `AUGVIS_ASSETS` | `stage4_record.py assets`, rendering | local mirror of the visual-augmentation assets |
| `DENSE_PLACE_REF_DIR` | rendering with place markers on (`DENSE_PLACE_MARKER`; a figure option) | folder of a frozen place-marker source that this repository does not ship (md5 checked by `record.render.markers_geom`); no default — a marker render dies without it |
| `ANTHROPIC_API_KEY` | `stage4_record.py text` | language-annotation requests (or the key file named in `dexengine/record/text/api.py`) |
| `DEXENGINE_POOL_ROOT` | placement judge, bake | reach-map root other than the canonical one (experiments only) |
| `DEXENGINE_PLUGINS` | RL tasks | comma-separated modules that register task hooks (`dexengine.rl.tasks.hooks`; used by `experiments/`) |

**Set by the launchers.** Users do not set these:

- `SIDE`, `GPU`, `SEQ_DIR`, `DEXENGINE_SEQ_LIST`, `FROZEN_REPLAY_*`, `RENDER_ONLY`, … are what the flags of
  `scripts/stage3_rl.py` become. A recorded launch environment can be replayed as is. `RENDER_ONLY` and
  `FROZEN_RENDER_OUT` only write dumps for a record session (with `DUMP_ONLY=1` or `DRYRUN=1`).
- The scene inputs of those dumps (`DEXENGINE_KEY_LIGHT`, `DEXENGINE_CAMERA_FOCAL_MM`) come from
  `dexengine.record.looks.scene_env`; the look itself is `dexengine.record.looks.look_d`. The knobs of the old render
  look (`DEXENGINE_AUG_*`, `DEXENGINE_LOG_EGO_PITCH`, `RENDER_EGO_FOCAL_MM`, `PRESET`) and the other removed renderer
  inputs (`dexengine.rl.render_removed`) die when set.
- The trainer-side `DEXENGINE_*` knobs come from the recipe.
- `DEXENGINE_SCENE` comes from `stage2_ground.sh --scene`.

To change one, use the flag or `--set` rather than exporting it by hand.

## Data layout

One data root holds everything that is not code. Paths are assembled only by `dexengine.core.layout`. Query them with
`python -m dexengine.core.paths --what layer --dataset <ds> --hand <hand> --emb <emb> --layer <layer>`.

| Stage | Reads | Writes |
|---|---|---|
| Data preparation | `data/<ds>/raw` · `data/body_models/mano` | `data/<ds>/mano/<emb>/<clip>` · `data/<ds>/assets/objects` · `data/<ds>/INTAKE` |
| 1 Retargeting | `mano` · objects · hand rig `assets/robots/hands/<spec>/mjcf` | `data/<ds>/retarget/<spec>/<emb>/<clip>` · `data/<ds>/base/<spec>/<emb>/<clip>` · `precompute/<ds>/{contact_mano_*, f0_stage, robots/<spec>}` |
| 2 IK grounding | `base` · f0 shadow · wrist cache · reach map `precompute/taco/ikpool` | `precompute/<ds>/robots/<combo>` · `data/<ds>/trees/<combo>/<emb>/<clip>_pl_rt` · `overfit/hands/<hand>/chunks/<tag>` |
| 3 RL | `trees` · arm-IK caches · robot and object USD | `logs/skrl/…` · `overfit/logs/refbase` · campaign cells `overfit/hands/<hand>/rl_campaigns` (in the [run tree](#data-root)) |
| 4 Recording | judged checkpoints · `trees` | `data/<ds>/collect/<combo>` (LeRobot) · `data/<ds>/collect/text` |

Run records (ledgers, audits, tables) go to `records/` of the run tree. For `xhand_pro`, `wuji2` and `dg5f_s` the
data root must be the code checkout: their vendor meshes live under `<data root>/data/_stage/`, Isaac's URDF importer
resolves meshes relative to the URDF in the code tree, and `scripts/build_robot_usd.py` refuses to bake when the two
differ ([`docs/hands.md`](hands.md)). Commands that take no `--data-root` read `DEXENGINE_MAIN_ROOT`
([`docs/configuration.md`](#data-root)).
Known limit: a data root that is itself a git checkout uses the code checkout as its run tree, so two such data roots
with one code checkout share `records/` and `overfit/`.

`<spec>` is the hand's retargeting asset folder (`--what hand-spec-dir --hand <hand>`). `<combo>` is
`<arm>_<hand>` (`--what combo-ids`). TACO keeps its hand-independent input layers elsewhere
(`dexengine.core.layout.STORAGE`).
