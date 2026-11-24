"""Dataset readers (raw capture -> MANO layer) and their registry.

``READERS`` has one row per dataset that has a reader in this package. A row says which driver runs the reader and,
for the per-recording ``stage`` driver (``python -m dexengine.data.read_raw``), the dataset-specific facts that
driver needs. Everything else (staging, resume, decompose, contact, retarget, ingest, counts) is the same for every
dataset. A dataset without a row is an error: there is no silent default. To add a reader, add its module under
``dexengine/data/readers/`` and one row here.

Row fields:

- ``module``: the reader module. The ``stage`` driver runs it once per recording as
  ``python -m <module> --dataset-dir <stage> --task <recording>``.
- ``driver``: ``stage`` (per-recording reader + ``read_raw``), ``hoi`` (library reader driven by
  ``dexengine.data.read_raw.hoi``), ``mano`` (raw dataset -> MANO layer with the shared CLI of
  ``readers.mano_raw.run_cli``: ``python -m <module> --raw <data/<ds>/raw> --mano-dir <dir> --out
  <data/<ds>/mano/bimanual> --all``; the module has ``enumerate_clips`` and ``convert``, checked against the
  delivered MANO layer by ``tests/data/verify_mano_readers.py``) or ``taco`` (TACO is read by the stage-1
  retarget run, ``dexengine.retarget``).
- ``tasks``: argv that prints the recordings, one per line (``{raw}`` = the dataset's raw layer, ``{pys}`` = the
  retargeting interpreter). Run from the repository root.
- ``links``: ``(stage-relative path, raw-relative path)`` pairs linked into the stage, where the reader looks by
  default.
- ``marker``: regex the reader prints once per recording (``{task}`` = the recording); the driver requires one per
  recording read.
- ``clip``: glob of the clip folders one recording leaves in the stage (``{task}``).
- ``read_args``: the ``mano`` reader's own arguments for the canonical MANO layer (h2o: ``--sides table``).
- ``downloads``: globs, relative to the dataset's raw layer, of the official download files — the only raw files the
  from-raw gate links into its empty root (``dexengine.release.fromraw``); a dataset without them cannot be gated.
- ``derive``: INTAKE derivations run in order once the canonical MANO layer and the object assets are in place, hand
  independent: ``(name, argv, text the log must contain or "")`` with ``{ds}`` and ``{out}``. The ``stage`` driver
  runs them after its ingest; for the other drivers ``python -m dexengine.data.read_raw <ds> derive`` runs them
  (``dexengine.data.read_raw.derive``). The base layer needs the support and segment audits; a row without them makes
  ``derive`` die. h2o's support audit rotates each clip by its desk plane, whose pivot reads the rest intervals of a
  support audit: its row unpacks the pose files, fits the frames, runs an untilted first audit, fits the planes, then
  the real audit. The depth frames of the planes are unpacked before (``dexengine.data.readers.h2o_unpack depth``).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from dexengine.core.errors import InputMissingError

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ReaderSpec:
    """One dataset's reader and the facts its driver needs (see the module docstring)."""

    module: str
    driver: str
    tasks: tuple[str, ...] = ()
    links: tuple[tuple[str, str], ...] = ()
    marker: str = ""
    clip: str = ""
    derive: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)
    downloads: tuple[str, ...] = ()
    read_args: tuple[str, ...] = ()


_SUPPORT = ("support_audit", "-m dexengine.data.support.plane --out {out}/support --datasets {ds} --write-intake", "")
_SEGMENT = ("segment_audit", "-m dexengine.data.segment.cuts --out {out}/segment --datasets {ds} --write-intake", "")

READERS: dict[str, ReaderSpec] = {
    "hot3d": ReaderSpec(
        module="dexengine.data.readers.hot3d",
        driver="stage",
        tasks=("python3", "-m", "dexengine.data.readers.hot3d_gt", "{raw}/dataset"),
        links=(("raw/hot3d", "dataset"),),
        marker=r"^\[HOT3D_MANO_LEFT\] task={task} .*left_flipped=[01]",
        clip="{task}__*",
        derive=(
            _SUPPORT,
            (
                "hot3d_desk_recover",
                "-m dexengine.data.support.hot3d --out {out}/hot3d_desk --write-intake",
                "'no_src': 0",
            ),
            _SEGMENT,
        ),
    ),
    "dexcanvas": ReaderSpec(
        module="dexengine.data.readers.dexcanvas",
        driver="stage",
        tasks=(
            "{pys}",
            "-m",
            "dexengine.data.readers.dexcanvas",
            "--list-only",
            "--parquet-path",
            "{raw}/hf/mocap_ver0.1.parquet",
        ),
        links=(("raw/dexcanvas/data", "hf"), ("raw/dexcanvas/objects", "toolkit/mano_assets/objects")),
        marker=r"^\[DEXCANVAS_EULER\] task={task} ",
        clip="{task}",
        derive=(_SUPPORT, _SEGMENT),
    ),
    "hocap": ReaderSpec(module="dexengine.data.readers.hocap", driver="hoi", derive=(_SUPPORT, _SEGMENT)),
    "hrdexdb": ReaderSpec(
        module="dexengine.data.readers.hrdexdb",
        driver="hoi",
        derive=(
            _SUPPORT,
            (
                "hrdexdb_desk_recover",
                "-m dexengine.data.support.hrdexdb --mode h1p --out {out}/hrdexdb_desk --write-intake",
                "[HRDEXDB_DESK] INTAKE updated",
            ),
            _SEGMENT,
        ),
    ),
    "taco": ReaderSpec(
        module="dexengine.data.readers.taco",
        driver="taco",
        derive=(_SUPPORT, _SEGMENT),
        downloads=("Hand_Poses/**/*",),
    ),
    "grab": ReaderSpec(module="dexengine.data.readers.grab", driver="mano", derive=(_SUPPORT, _SEGMENT)),
    "arctic": ReaderSpec(module="dexengine.data.readers.arctic", driver="mano", derive=(_SUPPORT, _SEGMENT)),
    "h2o": ReaderSpec(
        module="dexengine.data.readers.h2o",
        driver="mano",
        read_args=("--sides", "table"),
        downloads=("dl/subject*_pose_v1_1.tar.gz", "dl/object.zip"),
        derive=(
            ("h2o_unpack_poses", "-m dexengine.data.readers.h2o_unpack poses", "[H2O_UNPACK] poses files="),
            ("h2o_frame_fit", "-m dexengine.data.readers.h2o_frame --fit", "[h2o_frame] fit "),
            (
                "support_audit_untilted",
                "-m dexengine.data.support.plane --out {out}/support_untilted --datasets {ds} --write-intake "
                "--h2o-untilted",
                "[H2O_TILT] support_audit: untilted first pass",
            ),
            ("h2o_table_plane", "-m dexengine.data.support.h2o", "[h2o_table_plane]"),
            _SUPPORT,
            _SEGMENT,
        ),
    ),
    "oakink2": ReaderSpec(
        module="dexengine.data.readers.oakink2",
        driver="mano",
        derive=(
            _SUPPORT,
            (
                "oakink2_desk_recover",
                "-m dexengine.data.support.oakink2 --out {out}/oakink2_desk --write-intake",
                "[DESK_RULE] INTAKE updated",
            ),
            _SEGMENT,
        ),
    ),
}

#: The drivers a row may name (see the module docstring).
DRIVERS = ("stage", "hoi", "mano", "taco")


def spec(ds: str) -> ReaderSpec:
    """The reader row of `ds`; SystemExit names the registered datasets when there is none."""
    if ds not in READERS:
        raise InputMissingError(
            f"[REREAD] ✗ ds={ds}: no reader is registered (dexengine.data.readers.READERS · "
            f"registered {sorted(READERS)})"
        )
    return READERS[ds]


def stage_spec(ds: str) -> ReaderSpec:
    """The row of `ds` for the per-recording ``stage`` driver; SystemExit when its reader uses another driver."""
    s = spec(ds)
    if s.driver != "stage":
        raise SystemExit(
            f"[REREAD] ✗ ds={ds}: its reader runs under the '{s.driver}' driver, not the raw re-read "
            "stage (python -m dexengine.data.read_raw)"
        )
    return s


def list_tasks(ds: str, raw: str, pys: str) -> subprocess.CompletedProcess:
    """Run the recording lister of `ds` (its ``tasks`` argv) from the repository root; stdout = one recording per
    line. The return code is the lister's own."""
    argv = [a.format(raw=raw, pys=pys) for a in stage_spec(ds).tasks]
    return subprocess.run(argv, cwd=REPO, stdout=subprocess.PIPE, text=True)
