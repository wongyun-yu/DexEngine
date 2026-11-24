"""E5 GR00T policy modality JSON (schema ``e5_policy_modality/1``): what the checkpoint expects, composed from the
checkpoint processor / statistics, the dataset meta and a live server, every source cross-checked.

numpy only (no pyzmq / msgpack): ``policy_client.Gr00tClient`` loads one of these and the ``make-modality``
CLI of ``experiments/sim_eval/policy_client.py`` writes it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "e5_policy_modality/1"


class ModalityMismatch(ValueError):
    """Observation / server / modality JSON disagree on keys, dims, dtypes or horizons."""


# ----------------------------------------------------------------------------------------------- modality
@dataclass
class Modality:
    """What the checkpoint expects. Group order = column order of the concatenated state/action vector."""

    video_keys: list[str]
    image_hw: tuple[int, int]
    state_groups: list[tuple[str, int]]
    action_groups: list[tuple[str, int]]
    language_key: str
    action_horizon: int
    video_horizon: int = 1
    state_horizon: int = 1
    embodiment_tag: str = "new_embodiment"
    source: dict = field(default_factory=dict)

    @property
    def state_dim(self) -> int:
        return sum(d for _, d in self.state_groups)

    @property
    def action_dim(self) -> int:
        return sum(d for _, d in self.action_groups)

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA,
            "embodiment_tag": self.embodiment_tag,
            "video": {
                "keys": list(self.video_keys),
                "height": self.image_hw[0],
                "width": self.image_hw[1],
                "horizon": self.video_horizon,
            },
            "state": {"groups": [[k, d] for k, d in self.state_groups], "horizon": self.state_horizon},
            "action": {"groups": [[k, d] for k, d in self.action_groups], "horizon": self.action_horizon},
            "language": {"key": self.language_key},
            "source": self.source,
        }

    @classmethod
    def from_json(cls, j: dict) -> Modality:
        if j.get("schema") != SCHEMA:
            raise ModalityMismatch(f"modality schema {j.get('schema')!r} != {SCHEMA!r}")
        return cls(
            video_keys=list(j["video"]["keys"]),
            image_hw=(int(j["video"]["height"]), int(j["video"]["width"])),
            state_groups=[(str(k), int(d)) for k, d in j["state"]["groups"]],
            action_groups=[(str(k), int(d)) for k, d in j["action"]["groups"]],
            language_key=str(j["language"]["key"]),
            action_horizon=int(j["action"]["horizon"]),
            video_horizon=int(j["video"].get("horizon", 1)),
            state_horizon=int(j["state"].get("horizon", 1)),
            embodiment_tag=str(j.get("embodiment_tag", "new_embodiment")),
            source=dict(j.get("source", {})),
        )

    @classmethod
    def load(cls, src: Modality | dict | str | os.PathLike) -> Modality:
        if isinstance(src, Modality):
            return src
        if isinstance(src, dict):
            return cls.from_json(src)
        p = Path(src)
        if not p.is_file():
            raise FileNotFoundError(
                f"modality JSON {p} not found (build it with: python -m experiments.sim_eval.policy_client "
                "make-modality)"
            )
        return cls.from_json(json.loads(p.read_text()))

    # helpers for the Isaac side ------------------------------------------------------------------------
    def split_state(self, vec: np.ndarray) -> dict[str, np.ndarray]:
        """(D_state,) vector in group order -> {"state.<group>": (d,) float32}."""
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size != self.state_dim:
            raise ModalityMismatch(f"state vector has {v.size} values, modality wants {self.state_dim}")
        out, i = {}, 0
        for k, d in self.state_groups:
            out[f"state.{k}"] = v[i : i + d].copy()
            i += d
        return out

    def concat_action(self, action: dict[str, np.ndarray]) -> np.ndarray:
        """{"<group>": (T, d)} -> (T, D_action) in group order."""
        return np.concatenate([action[k] for k, _ in self.action_groups], axis=-1)


def _find_processor_dir(ckpt: Path) -> Path:
    # Gr00tPolicy (gr00t_policy.py L105-115): processor files at the root, else under processor/
    for d in (ckpt, ckpt / "processor", ckpt.parent / "processor"):
        if (d / "processor_config.json").is_file():
            return d
    raise FileNotFoundError(f"no processor_config.json in {ckpt}, {ckpt}/processor or {ckpt.parent}/processor")


def _stat_dim(stats_tag: dict, modality: str, key: str) -> int:
    try:
        s = stats_tag[modality][key]
    except KeyError as e:
        raise ModalityMismatch(f"statistics.json has no {modality}.{key}") from e
    dims = {len(np.asarray(v).reshape(-1)) for k, v in s.items() if k in ("min", "max", "mean", "std", "q01", "q99")}
    if len(dims) != 1 or 0 in dims:
        raise ModalityMismatch(f"statistics.json {modality}.{key} has inconsistent/empty dims {dims}")
    return dims.pop()


def _checkpoint_source(checkpoint, embodiment_tag, dims, src):
    """Keys/horizons from the checkpoint processor and dims from its statistics.json (fills dims and src)."""
    pdir = _find_processor_dir(Path(checkpoint))
    pc = json.loads((pdir / "processor_config.json").read_text())
    mcs = pc.get("processor_kwargs", pc)["modality_configs"]
    if embodiment_tag not in mcs:
        raise ModalityMismatch(f"checkpoint processor has no tag {embodiment_tag!r}; has {sorted(mcs)}")
    cfg = mcs[embodiment_tag]
    cands = [d / "statistics.json" for d in (pdir, Path(checkpoint))]
    st_path = next((c for c in cands if c.is_file()), None)
    if st_path is None:
        raise FileNotFoundError(f"no statistics.json next to {pdir / 'processor_config.json'}")
    st = json.loads(st_path.read_text())
    if embodiment_tag not in st:
        raise ModalityMismatch(f"statistics.json has no tag {embodiment_tag!r}; has {sorted(st)}")
    for m in ("state", "action"):
        dims[m] = {k: _stat_dim(st[embodiment_tag], m, k) for k in cfg[m]["modality_keys"]}
    src["checkpoint"] = str(Path(checkpoint).resolve())
    src["processor_config"] = str(pdir / "processor_config.json")
    src["statistics"] = str(st_path)
    return cfg


def _dataset_source(dataset, cfg, dims, image_hw, src):
    """Check the dataset meta against the model keys and dims; returns the image size (fills dims and src)."""
    meta = Path(dataset) / "meta"
    mj = json.loads((meta / "modality.json").read_text())
    ds_dims = {m: {k: int(v["end"]) - int(v["start"]) for k, v in mj[m].items()} for m in ("state", "action")}
    for m in ("state", "action"):
        spans = sorted(mj[m].items(), key=lambda kv: kv[1]["start"])
        if [k for k, _ in spans] != list(cfg[m]["modality_keys"]):
            raise ModalityMismatch(
                f"dataset {m} group order {[k for k, _ in spans]} != model {cfg[m]['modality_keys']}"
            )
        if m in dims and dims[m] != ds_dims[m]:
            raise ModalityMismatch(f"{m} dims checkpoint {dims[m]} != dataset {ds_dims[m]}")
        dims[m] = ds_dims[m]
    if sorted(mj["video"]) != sorted(cfg["video"]["modality_keys"]):
        raise ModalityMismatch(f"dataset video {sorted(mj['video'])} != model {cfg['video']['modality_keys']}")
    info = json.loads((meta / "info.json").read_text())
    hw = set()
    for v in mj["video"].values():
        f = info["features"][v["original_key"]]
        hw.add((int(f["info"]["video.height"]), int(f["info"]["video.width"])))
    if len(hw) != 1:
        raise ModalityMismatch(f"dataset cameras disagree on size {hw}")
    ds_hw = hw.pop()
    if image_hw is not None and tuple(image_hw) != ds_hw:
        raise ModalityMismatch(f"--image-hw {image_hw} != dataset {ds_hw}")
    image_hw = ds_hw
    src["dataset"] = str(Path(dataset).resolve())
    return image_hw


def modality_from_sources(
    *,
    checkpoint: str | None,
    dataset: str | None,
    server_cfg: dict | None,
    embodiment_tag: str,
    image_hw: tuple[int, int] | None,
) -> Modality:
    """Compose the modality JSON. Keys/horizons: checkpoint processor (or server). Dims: checkpoint statistics
    (or dataset meta/modality.json spans). Image size: dataset meta/info.json (or explicit). Every source given
    must agree with every other one, or this raises."""
    src: dict[str, Any] = {"embodiment_tag": embodiment_tag}
    cfg = None
    dims: dict[str, dict[str, int]] = {}
    if checkpoint:
        cfg = _checkpoint_source(checkpoint, embodiment_tag, dims, src)
    if server_cfg is not None:
        if cfg is None:
            cfg = server_cfg
        else:
            _compare_cfg(cfg, server_cfg, "checkpoint", "server")
        src["server_checked"] = True
    if cfg is None:
        raise ModalityMismatch("need --checkpoint or a live server for keys/horizons")
    if dataset:
        image_hw = _dataset_source(dataset, cfg, dims, image_hw, src)
    if not dims:
        raise ModalityMismatch("need --checkpoint (statistics.json) or --dataset (meta/modality.json) for dims")
    if image_hw is None:
        raise ModalityMismatch("image size unknown: give --dataset or --image-hw HxW (no default)")
    lang_keys = cfg["language"]["modality_keys"]
    return Modality(
        video_keys=list(cfg["video"]["modality_keys"]),
        image_hw=(int(image_hw[0]), int(image_hw[1])),
        state_groups=[(k, dims["state"][k]) for k in cfg["state"]["modality_keys"]],
        action_groups=[(k, dims["action"][k]) for k in cfg["action"]["modality_keys"]],
        language_key=lang_keys[0],
        action_horizon=len(cfg["action"]["delta_indices"]),
        video_horizon=len(cfg["video"]["delta_indices"]),
        state_horizon=len(cfg["state"]["delta_indices"]),
        embodiment_tag=embodiment_tag,
        source=src,
    )


def _compare_cfg(a: dict, b: dict, an: str, bn: str) -> None:
    for m in ("video", "state", "action", "language"):
        for f in ("modality_keys", "delta_indices"):
            if list(a[m][f]) != list(b[m][f]):
                raise ModalityMismatch(f"{m}.{f}: {an} {a[m][f]} != {bn} {b[m][f]}")
