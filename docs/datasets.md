# Datasets

DexEngine reads nine public hand–object datasets. None of them ships with this repository. Download each from its
source, accept its terms, and place it in the raw layer of your data root (`data/<ds>/raw`; for TACO,
`data/taco/raw_unzipped`). The terms of GRAB, ARCTIC and H2O do not allow redistribution.
Every dataset needs the MANO models (`MANO_RIGHT.pkl`, `MANO_LEFT.pkl`) from https://mano.is.tue.mpg.de in
`<data root>/data/body_models/mano/`, and the shipped input `generation`; TACO also needs three more shipped inputs
([Shipped inputs](#shipped-inputs)). Unpack them before running a reader.

The registries are the source of truth:

- `dexengine.core.layout.DATASETS`: capture rate, object id kind, embodiments, workspace.
- `dexengine.core.rules.profiles.DATASET_PROFILE`: judge scene, variant, object key.
- `dexengine.data.readers.READERS`: reader module and driver.

`python -m dexengine.core.paths --what datasets` lists them.

## Readers and drivers

A reader turns one dataset's raw files into the **MANO layer**: one folder per clip,
`data/<ds>/mano/<emb>/<clip>/{task_info.json, 0/trajectory_keypoints.npz}`. The object meshes (metres) belong in
`data/<ds>/assets/objects/<obj>/{visual.obj, convex/}`. The `stage` and `hoi` drivers write them; for the `mano`
readers, `dexengine.data.objects.build` builds them from the raw meshes after the reader has run (below). The
npz holds the world-frame wrist, fingertip and object poses of both hands as
`[x, y, z, qw, qx, qy, qz]`, plus wrist-local copies, resampled to the pipeline rate. The world is gravity-up (+z) and
grounded at the support surface. A reader runs under one of four drivers:

| Driver | Datasets | Command |
|---|---|---|
| `mano` | grab, arctic, h2o, oakink2 | `python -m dexengine.data.readers.<ds> --raw <data/<ds>/raw> --mano-dir <mano dir> --out <data/<ds>/mano/bimanual> --all` (GRAB adds `--subject-meshes <dir>`, H2O adds `--sides table`), then `python -m dexengine.data.objects.build <ds> --raw <data/<ds>/raw> --mano <data/<ds>/mano/bimanual> --out <data/<ds>/assets/objects> --workers <N>` |
| `stage` | hot3d, dexcanvas | `python -m dexengine.data.read_raw <ds> read`, then `python -m dexengine.data.read_raw <ds> intake <hand>` |
| `hoi` | hocap, hrdexdb | `python -m dexengine.data.read_raw.hoi --robot-type <spec> --dataset-root <stage> --datasets <ds>`, with `<stage>` laid out as below |
| `taco` | taco | read inside stage 1 (`scripts/stage1_retarget.py`, phases convert and kinematic) |

- **`mano` driver.** The shared code is in `dexengine/data/readers/mano_raw.py`: MANO forward kinematics, fingertips,
  grounding, resampling. `--list` prints the clip names.
- **Object assets of the `mano` datasets.** `python -m dexengine.data.objects.build` (retarget venv) writes
  `<obj>/visual.obj` and its convex parts `<obj>/convex/` for every object the clips of a MANO layer name (`--mano`),
  for listed names (`--objects a,b`) or for every object of the raw release (`--all`). Raw locations and asset names
  are the readers' own. `visual.obj` holds the raw vertex and triangle lists in raw order, in metres (ARCTIC's
  millimetres are divided by 1000), in the object frame; texture and normal indices are dropped, and no material or
  image is copied (rendering re-reads textures from the raw meshes, `dexengine/robots/textures/rules.py`). ARCTIC
  objects also get the part meshes `bottom.obj` and `top.obj`, and only the parts are decomposed
  (`convex_bottom/`, `convex_top/`). The parts come from the canonical decomposition
  (`dexengine/retarget/decompose/`). A rerun reuses equal files and current parts, and
  `--visual-only` skips the decomposition.
- **`stage` driver.** The reader runs per recording into a staging folder. Then decompose and contact detection run,
  the first hand is retargeted, mano and that hand are ingested, and the dataset's intake derivations run (row
  `derive`). `--dry` prints the decisions without writing.
- **`hoi` driver.** `<stage>` is any empty folder outside the data root (for example `<work>/hoi_<spec>`). The driver
  reads the raw data and the MANO models below it, not from
  `data/<ds>/raw` (`configs/data/read_hoi.yaml`), so link them there first:

  ```bash
  mkdir -p <stage>/raw/hocap <stage>/raw/hrdexdb <stage>/raw/taco/body_models
  ln -s <data root>/data/hocap/raw <stage>/raw/hocap/data          # HO-Cap: sequence_labels/, calibration/, models/
  ln -s <data root>/data/hrdexdb/raw <stage>/raw/hrdexdb/data      # HRDexDB: metadata/, human/, assets/
  ln -s <data root>/data/body_models/mano <stage>/raw/taco/body_models/mano   # MANO models (HO-Cap's fallback path)
  ```

  Each sequence runs through the reader and the retargeting chain (decompose, contact, scene, IK) for
  one hand into `--dataset-root`. A status log line per sequence makes it resumable. A run with no failure writes
  `processed/MANIFEST.sha256` for each dataset it read; admit the result with `python -m dexengine.data.intake
  <stage>/processed --hands <spec> --datasets <ds> --ingest --handoff-id <id>` (it refuses a stage without a manifest).
- **INTAKE derivations (`mano`, `hoi` and `taco` drivers).** The base layer needs `data/<ds>/INTAKE/support_audit.json`
  and `segment_audit.json`, audits of the canonical MANO layer. The `stage` driver makes them in its intake. For the
  other drivers run `python -m dexengine.data.read_raw <ds> derive` once the canonical MANO layer and the object assets
  are in place (`mano`: after the reader and `objects.build`; `hoi`: after the intake; `taco`: after the stage-1
  retarget). It runs the reader row's `derive` steps and fails unless both files were written with clips that are not
  all errors.
- **H2O is not buildable from the raw downloads in v0:** its support and segment audits need per-clip desk planes
  fitted from the RGB-D depth frames, and v0 has no step that unpacks those frames (`read_raw h2o derive` has no steps).
- **After any driver.** Retarget the other hands with `python -m dexengine.retarget.reretarget <ds> --hand <hand>`,
  then admit them with `python -m dexengine.data.intake` ([`pipeline.md`](pipeline.md), stage 1).

## Per-dataset notes

| Dataset | Reader | Raw files the reader expects | Notes |
|---|---|---|---|
| ARCTIC | `readers/arctic.py` | under `data/arctic/raw`: `data/arctic_data/data/raw_seqs/<sNN>/<seq>.{mano,object}.npy` and `data/arctic_data/data/meta/object_vtemplates/<obj>/{mesh,bottom,top}.obj` | Articulated objects with one hinge each. The object joint value is an extra qpos column, and the variant suffix is `_artic`. Redistribution not allowed. |
| DexCanvas | `readers/dexcanvas.py` | under `data/dexcanvas/raw`: `hf/mocap_ver0.1.parquet` (v0.1) and the toolkit's object meshes (`toolkit/mano_assets/objects`) | Right hand only. The hand-mesh vertex column is never loaded. |
| GRAB | `readers/grab.py` | the official zips `grab__s<N>.zip`, `tools__object_meshes__contact_meshes.zip` (read in place) | Needs the per-subject hand templates from GRAB's `tools/subject_meshes` (`--subject-meshes`), with no fallback. An object goes to every hand that touches it often enough (`CONTACT_RATIO`). Redistribution not allowed. |
| H2O | `readers/h2o.py` | `dl/subject<k>_pose_v1_1.tar.gz`, `dl/object.zip` (read in place, nothing extracted) | `--sides table\|contact` is required: `table` reproduces the delivered assignment in `configs/datasets/h2o/sides_delivered.json`, `contact` applies the reader's own contact rule. Redistribution not allowed. |
| HO-Cap | `readers/hocap.py` | under `data/hocap/raw` (linked into the `hoi` stage): `sequence_labels/<seq>.{meta.yaml, poses_m.npy, poses_o.npy}`, `calibration/mano.<subject>.yaml`, `models/<id>.cleaned_mesh_10000.obj` | Several objects per sequence: one attached object per hand, the rest as `extra_objects`. |
| HOT3D | `readers/hot3d.py` | under `data/hot3d/raw/dataset/` (the output folder of the official downloader): per sequence `<seq>/mano_hand_pose_trajectory.jsonl`, `dynamic_objects.csv`, `metadata.json`; `assets/<uid>.glb`, `assets/instance.json` (data groups `ground_truth` and `hand_data`; no VRS) | Room-scale workspace. Recordings are cut into clips by contact segmentation. The hand annotations are CC BY-NC-SA. |
| HRDexDB | `readers/hrdexdb.py` | under `data/hrdexdb/raw` (linked into the `hoi` stage): `metadata/episodes.parquet`, `human/<object>/<seq>/hand/mano_params/*.json`, object 6-D poses, `assets/mesh_v2/<object>/<object>.obj` | Only the human half is read. Right hand only. |
| OakInk2 | `readers/oakink2.py` | `hf/anno_preview/*.pkl`, `hf/program/program_info/*.json`, `hf/obj/object_repair/align_ds/<rid>/model.obj` | One clip per program stage. The y-up world is turned to z-up. Part pairs become articulated objects in base-layer step 2b. For textured objects the convex parts built from raw differ slightly (≤ 2.5 mm in extent, ±2 parts) from those used for the released clips, because the released meshes split vertices at texture seams. |
| TACO | `readers/taco.py` (+ `taco_convert.py`, `taco_lift.py`, `taco_source.py`) | `Hand_Poses/<triplet>/<seq>/{left,right}_hand{,_shape}.pkl`; the object tracks, meshes and convex parts are shipped inputs | Bimanual. See the note below. |

Download pages (accept each dataset's terms before downloading):

- TACO: https://taco2024.github.io
- ARCTIC: https://arctic.is.tue.mpg.de
- GRAB: https://grab.is.tue.mpg.de
- OakInk2: https://oakink.net/v2 (Hugging Face `kelvin34501/OakInk-v2`)
- HOT3D: https://facebookresearch.github.io/hot3d
- H2O: https://h2odataset.ethz.ch
- HO-Cap: https://irvlutd.github.io/HOCap
- HRDexDB: Hugging Face `HRDexDB/HRDexDB`
- DexCanvas: Hugging Face `DEXROBOT/DexCanvas`

**TACO.** The released TACO object tracks come from an earlier generation whose generator no longer exists; they are
not a resample of the public `Object_Poses` (same-time difference p99 12.2 mm, max 41.9°). The release therefore
ships them, with the TACO object meshes and convex parts: the shipped inputs `taco_source`, `taco_meshes` and
`taco_convex` ([Shipped inputs](#shipped-inputs) gives their folders and digests). Put the public `Hand_Poses` in
`data/taco/raw_unzipped`.
`python -m dexengine.data.readers.taco_source clips --out <list>` writes the clip list (every clip of the source).
`scripts/stage1_retarget.py --bases <list>` first runs a *lift* phase, which writes the raw TACO layout the
converter reads (`data/taco_retarget/raw/taco`: the hands shifted into the source frame, the source object tracks and
meshes), and then the batch phases. The public `Object_Poses` are not read anywhere: the converter dies on a layout
the lift did not write (no `*.grid50.npy` beside the poses), because converting them would give tracks that differ
from the released ones. Phase 1 reads the MANO models from `data/body_models/mano`.

**MANO left model.** The published `MANO_LEFT.pkl` has a known shape-direction defect. Whether a dataset's fits were
made with the corrected model is a property of the dataset. It is stated once, in
`dexengine/data/readers/mano_conventions.py` (`LEFT_FIX`), and readers and scene checks read it from there.

## Shipped inputs

Four intermediates cannot be rebuilt identically from the raw downloads, so the release ships them. Download each
from the [dataset release page](https://huggingface.co/datasets/Wongyun/DexEngine) and unpack it into its folder under the data root, together with its
checksum file `SHA256SUMS.<name>`, which goes into the same folder. The list, the folders and the digests live in
[`configs/release/shipped_inputs.tsv`](../configs/release/shipped_inputs.tsv):

| Input | Folder under the data root | Files | sha256 of `<folder>/SHA256SUMS.<name>` | Read by |
|---|---|---|---|---|
| `taco_source` | `data/taco/_generations/v1_o64/trajectories` | `*/*` | `2c24646db273cee2722a5cb3b5d7c35d31532cbe2afc9d8b3477b89b5985937f` | TACO stage 1 (lift) and the grasp reference |
| `taco_meshes` | `data/taco/meshes` | `*/visual.obj` | `53c3d51cbfbe4785826e6594914dbfdcb611643e25125a0bec03af296becede0` | TACO object surfaces |
| `taco_convex` | `data/taco_retarget/processed/taco/assets/objects` | `*/convex/*.obj` | `d69db71515f373a250cb499ed8b81971d81254531aeadb93317fe339062c1286` | TACO convex parts (stage 1 decompose) |
| `generation` | `data` | `<ds>/GENERATION`, one per dataset | `ddec344ebfb37e7690ea7547766336023230c2345a4027c042211e958c629c8a` | every dataset: the reference hand direction the dataset was built with (`dexengine/data/align.py`) |

- **The digest is that of the checksum file**, not of the download: `sha256sum <folder>/SHA256SUMS.<name>` prints it.
  The checksum file lists every file of the input with its sha256.
- **Each reader checks its input** and prints `[SHIPPED] <name> state=verified`: the checksum file has the registered
  digest and every listed file is present with its sha256 (a missing, changed or extra file stops the run). A folder
  without its checksum file is read as an unverified data root (`state=site`) and **no file is checked**, so
  always unpack the checksum file with the input. `python -m dexengine.core.shipped verify <name>` checks one input
  (rc 0 = verified).
- **`generation` is needed for every dataset.** Six datasets (arctic, dexcanvas, grab, h2o, hocap, hrdexdb) were
  built under another reference hand direction than the code's constant; without their declaration they stop with
  `[REF_DIR] ✗ <ds>: no generation declaration`. It unpacks to `data/<ds>/GENERATION` and
  `data/SHA256SUMS.generation`.
- **No from-raw gate in this version.** Unpack each input by hand as above; the readers check it where they read it.

## Intake of a delivered MANO or retargeting layer

A delivery that already holds the MANO layer, or retargeted clips, is admitted by
`python -m dexengine.data.intake <root> --datasets <ds> --hands <spec> [--ingest]`. The delivery must contain
`MANIFEST.sha256` and `<ds>/{mano, <hand>, assets/objects, assets/robots/<hand>}`. The hand asset's hash must be
registered in `dexengine.core.layout.RETARGET_HAND_ASSETS`. Every clip gets ACCEPT or REJECT. `--ingest` hard-links
only the accepted clips into the layout and writes the ledger `data/<ds>/INTAKE/<handoff_id>.json`. Clip ids come
from the delivered task names (`dexengine.core.layout.clip_id_from_task`).

## Adding a dataset

1. **Register it** in `dexengine.core.layout.DATASETS` with a `DatasetSpec`: source fps, object id kind, embodiments
   present, status, workspace (`tabletop` or `room`). Training entry points accept only datasets with status `canon`.
2. **Declare its profile** in `dexengine.core.rules.profiles.DATASET_PROFILE` and add the name to `profiles.ALL`.
   The profile sets the judge scene, the variant suffix, the stage-2 source layer, the object key used for bundling,
   and any campaign exclusion.
3. **Write a reader** `dexengine/data/readers/<ds>.py` and add its `ReaderSpec` row to
   `dexengine.data.readers.READERS`.
   - For the `mano` driver, the module provides `enumerate_clips`, `convert` and `main` through
     `mano_raw.run_cli`.
   - For the `stage` driver, the row gives the recording lister (`tasks`), the staging links, the per-recording
     marker, the clip glob and the intake derivations.
4. **State its MANO left-model convention** in `mano_conventions.LEFT_FIX` if the reader poses MANO parameters.
5. **Override the storage path** in `dexengine.core.layout.STORAGE`, but only if a hand-independent input layer has
   to live elsewhere.
6. **Run the registry tests**: `python -m pytest tests/core/test_registries.py tests/data/test_registries.py`. They
   check that every dataset has a reader, that the reader modules exist, and that each driver's contract holds.
7. **Run the pipeline.** Read the raw data, retarget (`reretarget <ds>`), then follow
   `dexengine/data/stages.py --status --datasets <ds>` through the base-layer steps, then cut the long-clip pieces
   and aerial tails with `ARM=<arm> CAMPAIGN_DIR=<log folder> python -m dexengine.data.stage1.split_apply <hand> --datasets <ds>`, then stage 2.

A clip name follows the grammar in `dexengine.core.paths` (`<task fields>__<take fields>`, normalised). Never build a
data path by hand: `dexengine.core.layout.render` assembles every layer path, and a dataset that is not registered
stops with a `KeyError`.
