# Running on a cluster

There are two layers:

- **Job backend** (`dexengine.cluster`). It runs any list of commands on local GPUs, or writes the list as Slurm or
  Kubernetes jobs. It knows nothing about the pipeline.
- **RL campaign runner** (`dexengine/rl/campaign.py`). It schedules the training and judging of many RL bundles on
  one GPU host.

Every job runs a command of the same checkout. Run jobs from a committed tree. Every node must see the data root at
the path the jobs name (`DEXENGINE_MAIN_ROOT`, [`configuration.md`](configuration.md)), and a node that runs Isaac Lab
jobs needs the Isaac Lab environment (the `isaac` extra).

## Job lists

A job list is a JSON Lines file with one `dexengine.cluster.jobs.Job` per line:

| Field | Meaning | Default |
|---|---|---|
| `name` | unique in the list; letters, digits, `.`, `_`, `-`; used for logs and scheduler names | required |
| `cmd` | argv of the command (no shell) | required |
| `env` | extra environment variables | `{}` |
| `gpus` | GPUs the command needs (0 = CPU only) | 1 |
| `cpus` | CPU cores | see `Job` |
| `mem_gb` | host memory | see `Job` |
| `time_min` | wall-time limit in minutes (0 = none) | 0 |
| `workdir` | working directory ("" = the caller's) | "" |
| `prep` | argv of a CPU step run before `cmd` (same env and workdir; must be safe to rerun), e.g. staging inputs | `[]` (none) |

Example: one RL bundle and one CPU retargeting run.

```json
{"name": "rl-grab-b01", "cmd": ["python", "scripts/stage3_rl.py", "grab_b01", "--side", "both", "--arm", "rby1", "--hand", "xhand_pro", "--dataset", "grab", "--emb", "bimanual", "--seq-dir", "data/grab/trees/rby1_xhand_pro/bimanual", "--seq-list", "lists/grab_b01.txt", "--gpu", "0"], "gpus": 1, "time_min": 720}
{"name": "retarget-grab", "cmd": ["python", "-m", "dexengine.retarget.reretarget", "grab", "16", "--hand", "xhand_pro"], "gpus": 0, "cpus": 16}
```

Unknown keys, duplicate names and negative resources are refused when the list is read. In Python, build lists with
`Job(...)` and write them with `dexengine.cluster.jobs.save_jobs`, which writes atomically. The CLI does not build
lists for you: write one job per bundle or per piece from your own bundle lists.

## Local GPUs: `run`

```bash
python -m dexengine.cluster run --jobs jobs.jsonl --gpus 0,1,2,3 --log-dir logs/jobs
```

- **Scheduling.** Jobs start in list order as soon as enough of the given GPUs are free. Each job sees only its own
  GPUs (`CUDA_VISIBLE_DEVICES`) and runs in its own session. A job that asks for more GPUs than were given is refused
  before anything starts.
- **Prep.** A job's `prep` runs first with no GPU (`CUDA_VISIBLE_DEVICES` empty) while other jobs hold the GPUs, so
  staging inputs does not keep a GPU idle. At most one prepared or preparing job waits per GPU; prepared jobs take
  the GPUs in list order. A failing prep is the job's exit code and its command does not run. Slurm and Kubernetes
  jobs run the prep inside their allocation (`job_entry`, an init container), so there the GPU is held during it.
- **Logs.** Output goes to `<log-dir>/<name>.log` and the exit code to `<log-dir>/<name>.rc`.
- **Resume.** A rerun skips every job whose `.rc` already holds `0`.
- **Stopping.** SIGTERM or SIGINT stops the running jobs, signalling each job's whole session.
- **Exit code.** 0 when every job exited 0, else 1.

## Slurm: `slurm`

```bash
python -m dexengine.cluster slurm --jobs jobs.jsonl --name rl --log-dir logs/jobs --out rl.sbatch --set partition=gpu
sbatch rl.sbatch
```

The output is one array script. Task `i` runs `python -m dexengine.cluster.job_entry --jobs <list> --index i`, which
replaces itself with job `i` and its environment. The job list must therefore be readable from the compute nodes. The
array requests the largest `gpus`, `cpus`, `mem_gb` and `time_min` among its jobs, so put jobs with very different
needs into separate lists.

Site settings come from [`configs/cluster/slurm.yaml`](../configs/cluster/slurm.yaml) (`SlurmConfig`):

- `partition`, `account`, `qos`;
- `python`: the interpreter on the nodes;
- `max_parallel`: the array throttle;
- `extra_sbatch`: extra `#SBATCH` lines;
- `setup`: shell lines run before the job, for example activating the Isaac Lab venv.

Override them with `--config FILE` or `--set key=value`.

## Kubernetes: `k8s`

```bash
python -m dexengine.cluster k8s --jobs jobs.jsonl --out rl.yaml --set image=<registry>/dexengine:<tag>
kubectl apply -f rl.yaml
```

The output has one `batch/v1` Job per list entry:

- The command and environment come from the entry.
- Resource requests equal the limits (`nvidia.com/gpu` when `gpus > 0`).
- `time_min` becomes `activeDeadlineSeconds`.

Settings come from [`configs/cluster/k8s.yaml`](../configs/cluster/k8s.yaml) (`K8sConfig`):

- `image` (required): a container image with Isaac Lab and this checkout;
- `namespace`, `name_prefix`, `service_account`, `image_pull_secret`, `node_selector`;
- `volumes`, as `{name: {pvc: claim, mount: path}}` or `{name: {host: path, mount: path}}`;
- `backoff_limit`, `ttl_seconds_after_finished`.

Job names are lower-cased and cut to the Kubernetes limit. Two entries that map to the same name are refused.

## RL campaign runner

`dexengine/rl/campaign.py` turns per-hand campaign manifests into trained, judged and finalised bundle cells.
`--manifests` points at a folder of `<hand>/manifest.json` files (built by `python -m dexengine.rl.manifest`, see
[`pipeline.md`](pipeline.md)). Each manifest lists the hand's bundles: id, class and clips. GPUs are named `<host>:<index,…|last>`. Without a host table the only host is this machine, labelled
`local` (`--judge-gpus last`, the default, is its last GPU). A site with several GPU machines names them in its layout
file (`DEXENGINE_LAYOUT_CONFIG`, see [`configuration.md`](configuration.md)): `hosts` maps a label to a hostname, and
`host_gpu_count_cmd` gives, per label, a shell command that prints that host's GPU count from another machine. A
campaign launches only on the host its GPU specs name.

```bash
DEXENGINE_ARM=rby1 python dexengine/rl/campaign.py --manifests <dir> --campaign <id> --order xhand_pro,wuji2 \
    --judge-gpus local:last --gpus local:0,1,2                                # dry run: plan only
DEXENGINE_ARM=rby1 python dexengine/rl/campaign.py --launch --manifests <dir> --campaign <id> --order xhand_pro,wuji2
```

- **Dry run** (the default). Plans the queue, estimates how long the training and judge GPUs take, and writes the
  plan (`--plan-out`).
- **`--launch`.**
  - Runs one loop per campaign on the GPU host. The loop is `dexengine/cluster/local.py` around the ticks of
    `dexengine/cluster/scheduler.py`.
  - It starts only from a committed tree, and when every hand's manifest is staged into its campaign cell
    (`--stage-hand`).
  - The judge GPUs are reserved for judges.
  - Each tick reads every bundle's state from the files in its cell and restarts a missing judge sidecar. It decides
    the overflow and finalises finished cells. It launches queued bundles on free training GPUs under the host gate
    (`scripts/rl/host_gate.sh`). A launch is a cold start: the cell and the work area must be empty.
- **Loop state.** `status.json`, `states.tsv` and `events.tsv` are written under the run layer
  `run_rl_loops/<campaign>`. A watcher reads `status.json` (`loop`: running, stopped, drained, ended, error,
  crashed). SIGTERM stops new launches; running trainings and sidecars continue in their own sessions.
- **After the campaign.** `dexengine/rl/cell_tools.py` relaunches failed bundles, renders the best and last
  checkpoints, and collects the per-hand result tables.

The judge sidecar's host gate reads the memory and concurrency limits `RLJUDGE_MIN_AVAIL_GB`, `RLJUDGE_RSS_GB` and
`RLJUDGE_MAX_CONC` (see [`configuration.md`](configuration.md)). Set them to match the host.
