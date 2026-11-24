"""Resume-safe raw -> retargeted trajectories for HRDexDB, HO-Cap and DexCanvas (driver ``hoi``).

Each selected sequence is read by its library reader (``list_sequences`` / ``load_sequence``), written as MANO
keypoints, and run through the retargeting chain (decompose, contact, scene, ik) for one hand rig. The output
must pass the same gate whether it is new or reused: the scene compiles, qpos is finite with the scene's width, and
no more than ``max_stuck_joints`` finger joints sit at a limit. One JSON line per sequence goes to the status log.
A run with no failure ends by writing ``processed/MANIFEST.sha256`` for every dataset it read
(``retarget.reretarget_stage.write_manifest``), which the intake verifies; a run with a failure leaves it as it was.

    python -m dexengine.data.read_raw.hoi --robot-type <hand spec dir> --dataset-root <stage root> \
        [--datasets hocap hrdexdb] [--task hocap:<task> ...|--smoke] [--config configs/data/read_hoi.yaml]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from dexengine.core import paths as P
from dexengine.core.repo import CONFIGS_DIR, REPO_ROOT
from dexengine.data.read_raw import derive
from dexengine.retarget.reretarget_stage import write_manifest

DATASETS = ("hrdexdb", "hocap", "dexcanvas")
DEFAULT_CONFIG = CONFIGS_DIR / "data" / "read_hoi.yaml"
RIG_FILES = ("right.xml", "left.xml", "bimanual.xml", "retarget_prior.yaml")
STUCK_TOL_DEG = 1.0  # a finger joint value within this of a limit counts as "at the limit"
STUCK_FRAC = 0.20  # ... for more than this fraction of the frames
STUCK_STD_DEG = 1.0  # ... while it barely moves (std below this) = a stuck joint


def _load_readers() -> dict:
    from dexengine.data.readers import dexcanvas, hocap, hrdexdb  # noqa: PLC0415  (heavy: MANO, mujoco)
    from dexengine.data.readers.common import write_sequence  # noqa: PLC0415

    return {"dexcanvas": dexcanvas, "hrdexdb": hrdexdb, "hocap": hocap, "write_sequence": write_sequence}


def load_config(path: Path) -> dict:
    """The driver config (schema 1, exactly the datasets in ``DATASETS``)."""
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    datasets = config.get("datasets", {})
    if set(datasets) != set(DATASETS):
        raise ValueError(f"{path}: datasets must be exactly {sorted(DATASETS)}, got {sorted(datasets)}")
    return config


def _run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def ensure_robot_assets(dataset_root: Path, dataset: str, robot_type: str) -> None:
    """Copy the hand rig into ``processed/<dataset>/assets/robots/<robot_type>`` (once) and check its files."""
    source = P.hand_rig_dir(robot_type)
    target = dataset_root / "processed" / dataset / "assets/robots" / robot_type
    if not source.is_dir():
        raise FileNotFoundError(f"robot asset is not built: {source}")
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"unexpected asset symlink: {target} -> {target.resolve()}")
        return
    if target.exists():
        if not target.is_dir():
            raise NotADirectoryError(target)
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    for name in RIG_FILES:
        if not (target / name).is_file():
            raise FileNotFoundError(f"incomplete robot asset: {target / name}")


def hocap_embodiment(sequence) -> str:
    """The embodiment of a sequence from the hands it has."""
    if sequence.right is not None and sequence.left is not None:
        return "bimanual"
    if sequence.right is not None:
        return "right"
    if sequence.left is not None:
        return "left"
    raise ValueError(f"{sequence.task}: no usable hand")


def _tasks_for(dataset: str, readers, dataset_root: Path, parquet_path: Path) -> list[str]:
    if dataset == "dexcanvas":
        return readers["dexcanvas"].list_sequences(parquet_path)
    return readers[dataset].list_sequences(dataset_root)


def _write_keypoints(
    dataset: str,
    task: str,
    readers,
    dataset_root: Path,
    parquet_path: Path,
    objects_dir: Path,
    target_ref_dt: float,
    ground: bool,
) -> tuple[str, str]:
    if dataset == "hrdexdb":
        sequence, embodiment = readers["hrdexdb"].load_sequence(dataset_root, task), "right"
    elif dataset == "hocap":
        sequence = readers["hocap"].load_sequence(dataset_root, task)
        embodiment = hocap_embodiment(sequence)
    elif dataset == "dexcanvas":
        sequence, embodiment = readers["dexcanvas"].load_sequence(parquet_path, objects_dir, task), "right"
    else:
        raise ValueError(dataset)
    readers["write_sequence"](
        sequence,
        dataset_root,
        dataset,
        embodiment,
        robot_type="mano",
        target_ref_dt=target_ref_dt,
        ground=ground,
        ground_mode="object_first_frame",
    )
    return embodiment, sequence.task


def stored_task(dataset: str, source_task: str) -> str:
    """The task folder name a source task is stored under."""
    return source_task.replace("/", "__") if dataset == "hrdexdb" else source_task


def trajectory_path(
    dataset_root: Path, dataset: str, robot_type: str, embodiment: str, task: str, data_id: int
) -> Path:
    """The retargeted trajectory of one task."""
    return (
        dataset_root
        / "processed"
        / dataset
        / robot_type
        / embodiment
        / task
        / str(data_id)
        / "trajectory_kinematic.npz"
    )


def validate_output(trajectory: Path, max_stuck_joints: int = 0) -> dict:
    """The output gate: scene compiles, qpos finite with the scene's width, stuck finger joints <= the limit."""
    import mujoco  # noqa: PLC0415

    scene = trajectory.parent.parent / "scene_eq.xml"
    if not scene.is_file():
        raise FileNotFoundError(f"missing generated scene: {scene}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(trajectory, allow_pickle=False) as archive:
        if "qpos" not in archive:
            raise KeyError(f"{trajectory}: missing qpos")
        qpos = archive["qpos"]
    if qpos.ndim != 2 or qpos.shape[0] < 2:
        raise ValueError(f"{trajectory}: invalid qpos shape {qpos.shape}")
    if qpos.shape[1] != model.nq:
        raise ValueError(f"{trajectory}: qpos width {qpos.shape[1]} != scene nq {model.nq}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"{trajectory}: qpos contains non-finite values")
    stuck = []
    tolerance = np.radians(STUCK_TOL_DEG)
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name or "finger" not in name or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        lower, upper = model.jnt_range[joint_id]
        values = qpos[:, model.jnt_qposadr[joint_id]]
        at_limit = np.mean((np.abs(values - lower) < tolerance) | (np.abs(values - upper) < tolerance))
        if at_limit > STUCK_FRAC and np.degrees(values.std()) < STUCK_STD_DEG:
            stuck.append(name)
    if len(stuck) > max_stuck_joints:
        raise ValueError(f"{trajectory}: {len(stuck)} STUCK finger joint(s), allowed {max_stuck_joints}: {stuck}")
    return {"frames": int(qpos.shape[0]), "nq": int(model.nq), "stuck_joints": len(stuck)}


def _process_one(dataset: str, task: str, readers, cfg: dict, env: dict[str, str]) -> tuple[str, str, Path, dict]:
    root, robot_type, data_id = cfg["dataset_root"], cfg["robot_type"], cfg["data_id"]
    embodiment, stask = _write_keypoints(
        dataset, task, readers, root, cfg["parquet"], cfg["objects"], cfg["target_ref_dt"], cfg["ground"]
    )
    ensure_robot_assets(root, dataset, robot_type)
    common = [
        "--dataset-dir",
        str(root),
        "--dataset-name",
        dataset,
        "--embodiment-type",
        embodiment,
        "--task",
        stask,
        "--data-id",
        str(data_id),
    ]
    py, pre = sys.executable, "dexengine.retarget."
    _run([py, "-m", pre + "decompose", *common], env)
    _run([py, "-m", pre + "contact", *common, "--no-show-viewer"], env)
    _run([py, "-m", pre + "scene", *common, "--robot-type", robot_type, "--no-show-viewer"], env)
    _run([py, "-m", pre + "ik", *common, "--robot-type", robot_type, "--no-show-viewer", "--seed", "0"], env)
    trajectory = trajectory_path(root, dataset, robot_type, embodiment, stask, data_id)
    return embodiment, stask, trajectory, validate_output(trajectory, max_stuck_joints=cfg["max_stuck_joints"])


def _status(path: Path, **record) -> None:
    record["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot-type", required=True, help="hand spec directory of the rig to retarget to")
    parser.add_argument("--dataset-root", type=Path, required=True, help="stage root (raw/ and processed/ below it)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS))
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        metavar="DATASET:TASK",
        help="Run only this item; may be supplied more than once.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run the one smoke_task declared for each dataset.")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--status-log", type=Path)
    args = parser.parse_args()
    if args.task and args.smoke:
        parser.error("--task and --smoke are mutually exclusive")
    if args.max_sequences < 0:
        parser.error("--max-sequences must be non-negative")
    if args.min_free_gb <= 0:
        parser.error("--min-free-gb must be positive")
    return args


def _select(args, config: dict, tasks_by_dataset: dict, selected_datasets: list) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    if args.task:
        for request in args.task:
            dataset, separator, task = request.partition(":")
            if not separator or dataset not in tasks_by_dataset or task not in tasks_by_dataset[dataset]:
                raise ValueError(f"unknown --task {request!r}; expected DATASET:TASK from the selected datasets")
            selected.append((dataset, task))
    elif args.smoke:
        for dataset in selected_datasets:
            task = str(config["datasets"][dataset]["smoke_task"])
            if task not in tasks_by_dataset[dataset]:
                raise ValueError(f"configured smoke task does not exist: {dataset}:{task}")
            selected.append((dataset, task))
    else:
        for dataset in selected_datasets:
            tasks = tasks_by_dataset[dataset]
            if args.max_sequences:
                tasks = tasks[: args.max_sequences]
            selected.extend((dataset, task) for task in tasks)
    return selected


def _resume(cfg: dict, dataset: str, task: str, index: str, status_log: Path) -> bool:
    """True when a valid earlier output of the task exists (recorded as skipped). An invalid one is reprocessed: a
    run killed mid-write leaves a truncated archive, which must not strand the task."""
    stask = stored_task(dataset, task)
    sides = ["right"] if dataset in {"hrdexdb", "dexcanvas"} else ["right", "left", "bimanual"]
    found = [
        (s, trajectory_path(cfg["dataset_root"], dataset, cfg["robot_type"], s, stask, cfg["data_id"])) for s in sides
    ]
    found = [(s, p) for s, p in found if p.is_file()]
    if len(found) > 1:
        raise RuntimeError(f"ambiguous resume outputs for {dataset}:{task}: {[str(p) for _, p in found]}")
    if not found:
        return False
    embodiment, candidate = found[0]
    try:
        stats = validate_output(candidate, max_stuck_joints=cfg["max_stuck_joints"])
    except Exception as exc:  # noqa: BLE001
        print(f"[{index}] existing output is not valid, reprocessing: {candidate} ({exc})", flush=True)
        return False
    print(f"[{index}] already complete: {candidate}")
    _status(
        status_log,
        dataset=dataset,
        task=task,
        stored_task=stask,
        embodiment=embodiment,
        state="skipped",
        trajectory=str(candidate),
        **stats,
    )
    return True


def main() -> int:
    args = _parse()
    config = load_config(args.config.resolve())
    trajectory = config["trajectory"]
    dataset_root = args.dataset_root.resolve()
    dex = config["datasets"]["dexcanvas"]
    cfg = {
        "dataset_root": dataset_root,
        "robot_type": args.robot_type,
        "data_id": int(trajectory["data_id"]),
        "target_ref_dt": float(trajectory["target_ref_dt"]),
        "ground": bool(trajectory["ground_on_first_object_frame"]),
        "max_stuck_joints": int(config["quality_gates"]["joint_audit"]["max_stuck_joints"]),
        "parquet": dataset_root / dex["parquet"],
        "objects": dataset_root / dex["objects"],
    }
    selected_datasets = args.datasets or list(config["datasets"])
    status_log = args.status_log or (dataset_root / "logs/retarget" / f"{args.robot_type}_hoi.jsonl")
    readers = _load_readers()
    tasks_by_dataset = {n: _tasks_for(n, readers, dataset_root, cfg["parquet"]) for n in selected_datasets}
    selected = _select(args, config, tasks_by_dataset, selected_datasets)
    print(
        f"Selected {len(selected)} sequences for {args.robot_type}: "
        + ", ".join(f"{d}={sum(n == d for n, _ in selected)}" for d in selected_datasets),
        flush=True,
    )
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    failures = 0
    for i, (dataset, task) in enumerate(selected, 1):
        index = f"{i}/{len(selected)}"
        output_parent = dataset_root / "processed" / dataset
        output_parent.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(output_parent).free
        if free_bytes < args.min_free_gb * 1024**3:
            raise RuntimeError(f"stopping before {dataset}:{task}: only {free_bytes / 1024**3:.1f} GiB free")
        print(f"[{index}] {dataset}:{task}", flush=True)
        try:
            if args.skip_existing and _resume(cfg, dataset, task, index, status_log):
                continue
            embodiment, stask, traj, stats = _process_one(dataset, task, readers, cfg, env)
            _status(
                status_log,
                dataset=dataset,
                task=task,
                stored_task=stask,
                embodiment=embodiment,
                state="complete",
                trajectory=str(traj),
                **stats,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[{index}] FAILED {dataset}:{task}: {exc}", file=sys.stderr, flush=True)
            _status(
                status_log,
                dataset=dataset,
                task=task,
                state="failed",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
    print(f"done: {len(selected)} selected, 0 excluded, {failures} failed (log {status_log})", flush=True)
    return finish(dataset_root, args.robot_type, selected, failures, status_log)


def finish(dataset_root: Path, robot_type: str, selected: list, failures: int, status_log: Path) -> int:
    """End of a run: any failure raises and leaves the manifest as it was; otherwise ``processed/MANIFEST.sha256``
    gets the lines of every dataset read (the intake verifies them) and the next steps are printed."""
    if failures:
        raise SystemExit(f"{failures}/{len(selected)} sequences failed; see {status_log} (manifest not written)")
    for dataset in sorted({d for d, _ in selected}):
        write_manifest(dataset_root, f"./{dataset}")
        print(
            f"next: python -m dexengine.data.intake {dataset_root / 'processed'} --hands {robot_type} "
            f"--datasets {dataset} --ingest --handoff-id <id>, then `{derive.command(dataset)}`",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
