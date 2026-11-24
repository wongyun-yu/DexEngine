# Third-party code, assets and data

This file records where the third-party code and assets in this repository come from, and under which terms. It is a
record, not legal advice. The repository's own code is under the Apache License 2.0
([`LICENSE`](LICENSE)); the terms of each third-party item take precedence over it for that item.

Binary assets (USD, meshes, textures) are stored with Git LFS (`.gitattributes`). Run `git lfs pull` after cloning.

## 1. Code in this repository

### Isaac Lab extension template (BSD-3-Clause)

Some files in `dexengine/rl/tasks/cfg/`, `dexengine/rl/tasks/mdp/` and `dexengine/core/artifact_paths.py` keep the
header "Copyright (c) 2022-2026, The Isaac Lab Project Developers … SPDX-License-Identifier: BSD-3-Clause". The
header comes from the Isaac Lab extension template the code started from. List them with
`grep -rl "SPDX-License-Identifier: BSD-3-Clause" dexengine`.

### Isaac Lab UrdfConverter settings (BSD-3-Clause)

`configs/robots/urdf_converter.yaml` holds UrdfConverter settings taken from an Isaac Lab asset's converter
config (`isaaclab_assets`), under the Isaac Lab BSD-3-Clause license.

### MV-Adapter (Apache-2.0)

`uv_parameterize_uvatlas` in `dexengine/robots/textures/raw_color.py` is ported from MV-Adapter
([`huanngzh/MV-Adapter`](https://github.com/huanngzh/MV-Adapter), `mvadapter/utils/mesh_utils/mesh_process.py`, the
function of the same name with the same arguments and defaults), licensed under the Apache License 2.0.

### skrl PPO update (MIT)

`dexengine/rl/train/klsafe_ppo.py` copies skrl 1.4.3's `PPO._update` with one change (the KL used for the early stop),
and `dexengine/rl/agents/per_side_ppo.py` overrides skrl internals (advantage, optimiser step, learning-rate update)
following the same code. skrl ([`Toni-SM/skrl`](https://github.com/Toni-SM/skrl)): Copyright (c) 2021 Toni-SM.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and
this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS
PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### SciPy Nelder-Mead arithmetic (BSD-3-Clause)

`NMPool` in `dexengine/retarget/f0/gpu_nm.py` reproduces the arithmetic of SciPy's `_minimize_neldermead`
(`scipy/optimize/_optimize.py`, adaptive coefficients, initial simplex) operation by operation, vectorised over
instances. SciPy: Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers. All rights reserved. Redistribution
and use in source and binary forms, with or without modification, are permitted provided that the following conditions
are met: (1) redistributions of source code must retain the above copyright notice, this list of conditions and the
following disclaimer; (2) redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided with the distribution;
(3) neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
products derived from this software without specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE
COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

### Geometry routines after Ericson, *Real-Time Collision Detection* (2005)

The segment–segment distance (`dexengine/placement/collision/geom.py`, §5.1.9) follows the book's routine, vectorised
with NumPy.

### Methods followed without their code

The retargeting stage follows the method of SPIDER ([`facebookresearch/spider`](https://github.com/facebookresearch/spider),
CC BY-NC 4.0) and the tracking rewards (terms and weights), terminations and PPO settings follow ManipTrans
([`ManipTrans/ManipTrans`](https://github.com/ManipTrans/ManipTrans), GPL-3.0). No code of either project is in this
repository: the implementation here is written independently from the published methods.

### Baselines and policy tools (`experiments/`)

| Item | Upstream | Terms | In this repository |
|---|---|---|---|
| DexMachina baseline | [`MandiZhao/dexmachina`](https://github.com/MandiZhao/dexmachina) and its simulator [`Genesis`](https://github.com/Genesis-Embodied-AI/Genesis) | MIT; Apache-2.0 | patch files (`experiments/baselines/dexmachina/patches/`) whose context lines are upstream code; the baseline runs from its own checkout |
| ManipTrans, SPIDER, FDR, CHORD baselines | their own repositories | their own licences | wrappers and input preparation only; each baseline runs from its own checkout |
| Weight EMA of the diffusion policy (`experiments/policies/dp/train_ema.py`) | [`real-stanford/diffusion_policy`](https://github.com/real-stanford/diffusion_policy) `EMAModel` | MIT | the same update rule and defaults, written for LeRobot |
| Policy-server wire format (`experiments/policies/dp/server.py`) | [`NVIDIA/Isaac-GR00T`](https://github.com/NVIDIA/Isaac-GR00T) policy server | Apache-2.0 | the message format only; no Isaac-GR00T code is imported |

## 2. Robot, hand and scene assets

A file the manufacturer publishes is not in this repository. `python -m dexengine.robots.vendor fetch` downloads it
from the pinned commit below and checks its sha256 ([`configs/robots/vendor/`](configs/robots/vendor) lists every file);
a file the manufacturer does not publish is shipped under [`assets/robots/vendor/`](assets/robots/vendor) and copied
the same way. The retargeting rigs, the merged robot URDFs and the robot USDs are built from these files by this
repository's code (`python -m dexengine.robots.vendor build`, `scripts/build_robot_usd.py`); they carry the terms of
the arm and hand they combine.

| Asset | Source | In this repository | Terms |
|---|---|---|---|
| RB-Y1 arm and body URDF and meshes | Rainbow Robotics [`rby1-sdk`](https://github.com/RainbowRobotics/rby1-sdk) `models/rby1a/urdf/`, commit `60a97d6c` | no (fetched) | Apache-2.0 |
| Sharpa Wave hand URDF, MJCF and meshes | Sharpa [`sharpa-urdf-usd-xml`](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) `wave_01/`, commit `6eea427e` | no (fetched, with that repository's `LICENSE.txt` and `NOTICE.txt`) | Apache-2.0 |
| Tesollo DG-5F-S hand URDF and meshes | Tesollo [`dg5f_s_ros2`](https://github.com/tesollodelto/dg5f_s_ros2) `dg5f_s_description/`, commit `284501f7` | no (fetched) | BSD-3-Clause |
| Wuji Hand 2 hand meshes | Wuji Technology [`wuji-description`](https://github.com/wuji-technology/wuji-description) `hand2/hand2_beta2/body/`, release v2026.8.14 (`06e5f14c`) | no (fetched) | MIT |
| Wuji Hand 2 URDFs as mounted on RB-Y1, the adapter parts between the arm flange and the hand, and the wrist frames | the hardware setup; hand kinematics and inertia from `wuji-description` | yes, `assets/robots/vendor/wuji2/`, with `wuji-description`'s MIT notice in `assets/robots/vendor/wuji2/LICENSE` | MIT for the vendor hand data (kinematics, inertia, meshes); repository licence for the adapter parts, the wrist frames and the mount |
| XHand Pro hand URDF and meshes | a third-party public repository, [`lixiaolong04/rotate_dexgen`](https://github.com/lixiaolong04/rotate_dexgen) `assets/xhand_green/`, commit `28a7a242` (not the maker's release; byte-identical to the files this project used) | no (fetched) | no separate notice for these files; the repository `LICENSE` at that commit applies: Apache-2.0, Copyright 2025 Sharpa Group (sha256 `ecb5a318c68e023cc889f11232c43c1e3a865cfd29a4359043229f47a1392c2b`) |
| ZED Mini camera mesh (head camera of every robot body and of the render body) | Stereolabs [`zed-ros2-description`](https://github.com/stereolabs/zed-ros2-description) `meshes/zedm.stl`, commit `449eef95` | no (fetched) | Apache-2.0 |
| Arm collision proxies, scene blocks, rig placements (`retarget_prior.yaml`), the head camera link (`assets/robots/head_cameras/zed_camera_link.xml`: placement, and the ZED Mini datasheet mass) | this project | yes | repository licence |
| Willow desk | not in the repository: `python -m dexengine.robots.vendor fetch` (the same as `python -m dexengine.record.looks.look_d --mirror`) downloads `Assets/ArchVis/Commercial/Tables/Willow.usd` and its 7 textures from NVIDIA's S3 into the local look mirror; the code pins their sha256 (`dexengine.record.looks.look_assets.WILLOW_SHA256`) | no | **NVIDIA terms: not redistributed — each user downloads it from NVIDIA under NVIDIA's terms.** Every scene uses it; a missing or changed file stops the run (`look_assets.willow_usd`) |

## 3. Assets fetched at run time (not in the repository)

| What | Fetched by | Terms |
|---|---|---|
| Scene materials (walls, floor MDL) | Isaac Sim / Kit, from NVIDIA's asset server | NVIDIA Isaac Sim Additional Software and Materials License: not redistributed — each user fetches them under NVIDIA's terms |
| Visual-augmentation textures, models and HDRIs | `python scripts/stage4_record.py assets` into `AUGVIS_ASSETS` | Poly Haven (CC0), ambientCG (CC0), Google Scanned Objects (CC BY 4.0); the asset list kept in `configs/record/augvis_v2/` (installed as the root's `manifest.json`) records each asset's licence and source page, and every downloaded file is checked against its sha256 there |

## 4. Datasets and body models (not in the repository)

No source dataset and no MANO model is distributed here. Download each from its source and accept its terms yourself
([`docs/datasets.md`](docs/datasets.md) lists the download pages). Data derived from a dataset follows that dataset's
terms: the MANO layer, the retargeted trees and the recorded LeRobot datasets. The tree tracks these small
dataset-derived files, each under the terms of its dataset:

| File | Derived from | Content |
|---|---|---|
| `configs/datasets/oakink2_lever_rigid.tsv`, `configs/datasets/artic_ranges.json` | OakInk2 | per-clip part offsets and measured articulation ranges |
| `dexengine/data/oakink2/part_kinds.py` | OakInk2 | the part ids of the objects and their parent parts (a table in code) |
| `configs/datasets/h2o/sides_delivered.json` | H2O | which hand holds the object in each take |
| `configs/datasets/*/excluded_clips.txt` | each dataset | clip names left out, with the reason |

| Dataset or model | Terms |
|---|---|
| GRAB | redistribution not allowed; also needs the per-subject hand templates from GRAB's `tools/subject_meshes` |
| ARCTIC | redistribution not allowed |
| H2O | redistribution not allowed |
| HOT3D | hand data (MANO and UmeTrack poses) CC BY-NC-SA 4.0; sequences and other annotations CC BY-SA 4.0; 3D object models CC BY-SA 4.0 with a no-sale clause ([licence](https://www.projectaria.com/datasets/hot3D/license/)). Robot data derived from the hand poses follows CC BY-NC-SA. |
| TACO | CC BY 4.0 (TACO README, github.com/Satorinal/TACO); attribution: TACO, Liu et al., CVPR 2024. The release ships three TACO-derived inputs the public code cannot rebuild (`configs/release/shipped_inputs.tsv`): `taco_source` — object trajectories and grasp assignment from an earlier processing of the official `Object_Poses` (world frame lowered to the table per capture session, resampled to the 50 Hz output grid; at the same timestamps they differ from the official tracks by p99 12.2 mm, and that processing is not recoverable), `taco_convex` — convex parts of the object meshes (CoACD), `taco_meshes` — the object meshes in metres |
| OakInk2 | CC BY-NC-SA 3.0 (the authors' dataset card on Hugging Face, `kelvin34501/OakInk-v2`; the project site states none) |
| HO-Cap | CC BY 4.0 |
| HRDexDB | CC BY-NC 4.0 (dataset card, `JongbinLim/HRDexDB`) |
| DexCanvas | ODbL 1.0 (dataset card, `DEXROBOT/DexCanvas`): a derived database is shared under the same licence |
| MANO hand model (`MANO_RIGHT.pkl`, `MANO_LEFT.pkl`) | MANO licence ([mano.is.tue.mpg.de](https://mano.is.tue.mpg.de)): non-commercial research use, not redistributable — each user downloads it; place it in `<data root>/data/body_models/mano/` |

## 5. Runtime dependencies

Isaac Sim, Isaac Lab, PyTorch, skrl, MuJoCo, pinocchio, smplx and the other packages in `pyproject.toml` are installed
separately and keep their own licences.
