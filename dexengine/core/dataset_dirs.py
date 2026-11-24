"""Training directories of a dataset: `resolve(name)` → `DatasetSpec` with absolute paths.

The RL entry point accepts `--dataset <name>` and fills `--data_dir`, `--object_mesh_dir` (and the unused
held-out / object USD slots) from here; explicit flags still win. Every path is built by `dexengine.core.layout`
(the base layer of the axis hand and the dataset's meshes layer), rooted at the data root.
"""

from __future__ import annotations

from dataclasses import dataclass

from dexengine.core import layout as _layout


@dataclass(frozen=True)
class DatasetSpec:
    """Absolute training paths of one dataset."""

    name: str
    train_data_dir: str | None  # base layer of the axis hand (representative embodiment)
    heldout_data_dir: str | None  # None: no dataset has a held-out split
    object_usd: str | None  # None: the tree's objects[] name the objects
    object_mesh_dir: str | None  # meshes layer (surface_points.npz)
    notes: str = ""


def _entry(name: str, spec) -> dict:
    emb = "bimanual" if "bimanual" in spec.embodiments else spec.embodiments[0]
    train = _layout.render("base", name, hand=_layout.axis_spec_dir(), emb=emb) if spec.status == "canon" else None
    return {
        "train_data_dir": train,
        "heldout_data_dir": None,
        "object_usd": None,
        "object_mesh_dir": _layout.render("meshes", name),
        "notes": f"status={spec.status}: {spec.notes}",
    }


_REGISTRY_RAW: dict = {name: _entry(name, spec) for name, spec in _layout.DATASETS.items()}


def _abspath(p: str | None) -> str | None:
    return None if p is None else _layout.data_abspath(p)


def resolve(name: str) -> DatasetSpec:
    """The training paths of dataset `name`. KeyError for an unknown dataset or one that is not canon yet."""
    if name not in _REGISTRY_RAW:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(_REGISTRY_RAW)}.")
    raw = _REGISTRY_RAW[name]
    if raw["train_data_dir"] is None:
        raise KeyError(
            f"Dataset '{name}' is registered but not trainable yet "
            f"(status={_layout.DATASETS[name].status}): no base tree."
        )
    return DatasetSpec(
        name=name,
        train_data_dir=_abspath(raw["train_data_dir"]),
        heldout_data_dir=_abspath(raw["heldout_data_dir"]),
        object_usd=_abspath(raw["object_usd"]),
        object_mesh_dir=_abspath(raw["object_mesh_dir"]),
        notes=raw["notes"],
    )
