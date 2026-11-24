# Installation

DexEngine runs from a checkout (the files under `scripts/` put the checkout on `sys.path`; set `PYTHONPATH=<checkout>`
for `python -m …` and for the files under `dexengine/` run by path). The package metadata includes `dexengine/` only,
while the code reads `configs/`, `assets/` and `scripts/` beside it, so always run from the checkout rather than from an
installed copy. Binary assets are stored with Git LFS:

```bash
git lfs install && git lfs pull
```

Two Python environments are used:

| Environment | Used by | Install |
|---|---|---|
| Retargeting venv (Python 3.12 was used) | stage 1, the CPU steps of stage 2, recording orchestration, tests | `python -m venv .venv && .venv/bin/pip install -e ".[retarget,test]"` |
| Isaac Lab environment | placement judge, stage 3, rendering, robot USD bake | Isaac Lab's own venv (below), then `uv pip install -e ".[isaac]"` in it |

System programs: `git-lfs`, and `zstd` and `nice` on `PATH` for recording (the depth frames are written through the
`zstd` executable, `dexengine/record/render/writer.py`).

The Isaac Lab environment as tested: Isaac Lab's pip installation with its `uv` venv. The steps follow Isaac Lab's
`isaaclab.sh` and `docs/source/setup/installation/pip_installation.rst` at that commit (Linux x86_64); they were not
re-run for this release.

```bash
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab && git checkout 88d94ea82b
./isaaclab.sh --uv                    # creates ./env_isaaclab (the venv name the scripts look for)
source env_isaaclab/bin/activate
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
./isaaclab.sh --install skrl          # the Isaac Lab extensions and skrl
cd <checkout> && uv pip install -e ".[isaac]"
export ISAACLAB_DIR=<path to IsaacLab>
```

- Both extras pin MuJoCo 3.4.0. The recorded retargeting runs used a MuJoCo development build
  (`3.4.1.dev861077626`) that no package index serves; 3.4.0 is the nearest release. Under 3.4.0 the committed rigs
  compile to the same models and forward kinematics and `hand_rig.py --dry-run` reports the same, but full
  retargeting outputs were not regenerated with it, so they can differ from the recorded ones. The recorded
  environment is [`configs/env/retarget_requirements.txt`](../configs/env/retarget_requirements.txt) (a pip freeze, not
  an install input).
- The retargeting extra also pins `warp-lang` (the f0 GPU engine imports warp), `mujoco-warp`, `usd-core` (step 4
  reads object USDs), `threadpoolctl` (CoACD decomposition) and `opencv-python` (the stage-4 `assets` tool); the
  reasons are stated next to each pin in `pyproject.toml`. `tests/retarget/test_retarget_extra.py` keeps every package that code run by the retargeting
  interpreter imports in that extra.
- The retargeting interpreter is `RETARGET_PY` when set, else the code checkout's `.venv/bin/python` (the venv
  belongs to the code, not to the data root, which is never searched). A set `RETARGET_PY` that does not exist is an
  error, not a reason to use `.venv`. `python -m dexengine.core.paths --what retarget-py` prints the resolved path.
- Isaac Lab is found at `ISAACLAB_DIR`. Unset, the only fallback is a sibling `../IsaacLab` next to the code
  checkout (also next to the working directory, the data root and the main checkout of a git worktree). A tree
  installed or exported anywhere else must set `ISAACLAB_DIR`; otherwise the first Isaac session stops with
  `[isaac_py] IsaacLab not found — set ISAACLAB_DIR`. Expected layout: `$ISAACLAB_DIR/isaaclab.sh` and an executable
  `$ISAACLAB_DIR/env_isaaclab/bin/python` (the Isaac Lab venv; rule: `scripts/_isaac_py.sh`). The tested environment is Isaac Lab at commit `88d94ea82b`
  (`git describe`: `v2.2.1-226`, extension `isaaclab` 0.53.1), Isaac Sim 5.1.0 (pip), Python 3.11, PyTorch 2.7.0
  (CUDA 12.8) and skrl 1.4.3. The code does not check these versions.
- MANO models (`MANO_RIGHT.pkl`, `MANO_LEFT.pkl`) are not distributed here; download them from the MANO website and
  place them in `<data root>/data/body_models/mano/`.
- Shipped inputs: four intermediates the public code cannot rebuild identically (the reference hand direction of
  every dataset, `generation`, and three TACO inputs). Download them from the [dataset release page](https://huggingface.co/datasets/Wongyun/DexEngine)
  and unpack each with its `SHA256SUMS.<name>` file into its folder under the data root before running a reader;
  folders and digests: [`docs/datasets.md`](datasets.md#shipped-inputs).

## Robot, hand and scene assets

The repository carries no file a manufacturer publishes. One command downloads every robot and hand file the builders
read from the manufacturer's public repository at a pinned commit, checks its sha256 and places it (a file the
manufacturer does not publish is shipped under [`assets/robots/vendor/`](../assets/robots/vendor) and copied the same
way); it then builds the scene look mirror (NVIDIA materials and the Willow desk, which are not redistributable) when
it is missing. A second command builds the retargeting rigs and the merged robot URDFs from those files:

```bash
python -m dexengine.robots.vendor fetch --data-root <data root>   # robots, hands, the scene look mirror
python -m dexengine.robots.vendor check --data-root <data root>   # every listed file present with its sha256
python -m dexengine.robots.vendor build --data-root <data root>   # hand rigs -> merged robot URDFs -> limit variants
python scripts/build_robot_usd.py                                 # robot USDs (Isaac Lab environment, GPU)
```

A run reads only the hand it uses: `fetch --only <sources>` and `build --hands <hand>` are enough, with these source
lists (`look_mirror` is the scene desk and materials, which every scene needs):

| Hand | `fetch --only` |
|---|---|
| `xhand_pro` | `rby1,zed_mini,xhand_pro,look_mirror` |
| `sharpa_wave` | `rby1,zed_mini,sharpa_wave,look_mirror` |
| `dg5f_s` | `rby1,zed_mini,dg5f_s,look_mirror` |
| `wuji2` | `rby1,zed_mini,wuji2,wuji2_body,look_mirror` |

For `xhand_pro`, `wuji2` and `dg5f_s`, `<data root>` is the code checkout ([Data layout](configuration.md#data-layout)).

With every hand fetched and built, the arm's reach statement is also checked against every hand
(`[TORSO_REACH]`); otherwise it is used as stated. A file a repository keeps in Git LFS is served as a pointer at the raw URL: a pointer whose oid is the listed
sha256 is followed to the LFS object at the same commit. A file whose bytes differ from the listed sha256 stops the fetch before it is written. A data root that already
holds generated data (robot USDs, trees) is refused unless `--allow-data-root` is given.

**XHand Pro: from a third-party public repository.** The maker does not publish these files. `xhand_pro` in
`sources.yaml` pins [`lixiaolong04/rotate_dexgen`](https://github.com/lixiaolong04/rotate_dexgen) at `28a7a242`
(`assets/xhand_green/xhand_{left,right}`): its two URDFs and 75 meshes per side are byte-identical to the files this
project was built with (sha256 in `files.tsv`). That repository states no separate terms for these files (its
LICENSE is Apache-2.0, Copyright Sharpa Group). This repository does not redistribute them; `fetch` downloads them.

The sources, pinned commits and licences are listed in [`configs/robots/vendor/sources.yaml`](../configs/robots/vendor/sources.yaml),
every file with its sha256 in [`files.tsv`](../configs/robots/vendor/files.tsv) beside it; [`THIRD_PARTY.md`](../THIRD_PARTY.md)
gives the terms. Files under `data/` go to the data root, the others into the checkout (ignored by git).

## Tests

Run the tests after the asset steps above (`vendor fetch` and `vendor build`); without the fetched robot and hand files the test session stops at start-up.

```bash
.venv/bin/python -m pytest            # from the checkout; every test gets an empty temporary data root
```

Tests that need `torch`, `mujoco` or a real data root (`DEXENGINE_MAIN_ROOT=<root>`) skip themselves otherwise.
