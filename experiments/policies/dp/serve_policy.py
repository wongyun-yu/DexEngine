"""The DP policy behind the server: observation → absolute joint targets, and the oracle that returns training labels
through the same path."""

from __future__ import annotations

import glob
import json
import time

import numpy as np
import pandas as pd
import torch

from experiments.policies.dp.serve_model import (
    CAMS,
    GROUPS,
    LANG,
    SCHEMA,
    const_state_dims,
    load_policy,
    oracle_label_source,
    stats_check,
)


class DPServe:
    """LeRobot diffusion policy with the GR00T server endpoints (get_action, reset, get_modality_config)."""

    def _task_axes(self, root):
        """task_index → axis_index over every training parquet; a task on two axes dies."""
        t2a = {}
        for f in sorted(glob.glob(f"{root}/data/chunk-*/*.parquet")):
            d = pd.read_parquet(f, columns=["task_index", "axis_index"])
            for t, a in set(zip(d["task_index"].tolist(), d["axis_index"].tolist(), strict=True)):
                if t2a.setdefault(t, a) != a:
                    raise SystemExit(f"task_index {t} maps to two axes ({t2a[t]}, {a}) in {root}")
        return t2a

    def __init__(
        self,
        ckpt: str,
        root: str,
        device: str,
        oracle_map: str = "",
        oracle_log: str = "",
        oracle_root: str = "",
        sampler: tuple[str, int] | None = None,
        const_state_fix: bool = False,
        const_state_eps: float = 0.0,
        oracle_exact: bool = False,
        oracle_hold_end: bool = False,
    ):
        self.dev = torch.device(device)
        self.p, c = load_policy(ckpt, oracle_map, self.dev, sampler)
        self.ms = [] if sampler is not None else None  # sampler override: generate_actions wall times (ms)
        self.n_obs, self.n_act = c.n_obs_steps, c.n_action_steps
        self.img_keys = list(c.image_features)
        want = {f"observation.images.{k}" for k in CAMS}
        if set(self.img_keys) != want:
            raise SystemExit(f"checkpoint image features {self.img_keys} ≠ {sorted(want)}")
        with open(f"{root}/meta/tasks.jsonl") as fh:
            tasks = {json.loads(x)["task"]: json.loads(x)["task_index"] for x in fh}
        t2a = self._task_axes(root)
        self.text2axis = {txt: t2a[ti] for txt, ti in tasks.items() if ti in t2a}
        self.stats_sha = stats_check(self.p, root, ckpt)
        self.const_dims, self.const_min, self.const_mark = None, None, ""
        if const_state_eps and not const_state_fix:
            raise SystemExit("--const-state-eps needs --const-state-fix")
        if const_state_fix:
            self.const_dims, self.const_min, self.const_mark = const_state_dims(self.p, ckpt, const_state_eps)
            print(self.const_mark, flush=True)
        with open(f"{root}/meta/info.json") as fh:
            info = json.load(fh)
        self.env_dim = int(info["features"]["observation.environment_state"]["shape"][0])
        # groups · cameras · image size: the training dataset's meta must say what GROUPS · CAMS say (else die)
        with open(f"{root}/meta/modality.json") as fh:
            mj = json.load(fh)
        for kind in ("state", "action"):
            got = tuple((g, int(v["start"]), int(v["end"])) for g, v in mj[kind].items())
            if got != GROUPS:
                raise SystemExit(f"{root}/meta/modality.json {kind} {got} ≠ server groups {GROUPS}")
        if sorted(mj["video"]) != sorted(CAMS) or list(mj.get("annotation", {})) != [LANG]:
            raise SystemExit(
                f"{root}/meta/modality.json video {sorted(mj['video'])} · annotation "
                f"{list(mj.get('annotation', {}))} ≠ {sorted(CAMS)} · [{LANG}]"
            )
        shapes = {tuple(info["features"][f"observation.images.{k}"]["shape"]) for k in CAMS}
        if len(shapes) != 1:
            raise SystemExit(f"camera shapes differ {shapes}")
        _c, self.H, self.W = next(iter(shapes))
        self.ckpt, self.root, self.horizon = ckpt, root, c.horizon
        self.omap, self.olog = None, oracle_log
        self.exact = oracle_exact
        lroot = oracle_label_source(root, oracle_map, oracle_root, oracle_exact)
        if oracle_hold_end and not oracle_map:
            raise SystemExit("[E5_DP_ORACLE_HOLD_END] needs --oracle-map")
        self.hold_end = oracle_hold_end
        if oracle_hold_end:
            print(
                "[E5_DP_ORACLE_HOLD_END] on · queries past the oracle episode end repeat its last label "
                "(tail = frames past)",
                flush=True,
            )
        if oracle_map:
            self._load_oracle(oracle_map, lroot, c)
        self.n_inf = c.num_inference_steps
        print(
            f"[E5_DP_SERVER] loaded {ckpt} n_obs={self.n_obs} horizon={c.horizon} n_action={self.n_act} "
            f"cams={self.img_keys} env_dim={self.env_dim} tasks→axis={self.text2axis}",
            flush=True,
        )

    def _load_oracle(self, oracle_map: str, lroot: str, c) -> None:
        with open(oracle_map) as fh:
            self.omap = json.load(fh)
        want = {int(v["episode"]) for v in self.omap.values()}
        acts, sts = {}, {}
        for f in sorted(glob.glob(f"{lroot}/data/chunk-*/*.parquet")):
            d = pd.read_parquet(f, columns=["episode_index", "frame_index", "action", "observation.state"])
            for e, g in d.groupby("episode_index"):
                if int(e) in want:
                    g = g.sort_values("frame_index")
                    acts[int(e)] = np.stack(g["action"].to_numpy()).astype(np.float32)
                    sts[int(e)] = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        for sc, v in self.omap.items():
            e = int(v["episode"])
            if e not in acts or len(acts[e]) != int(v["length"]):
                n = len(acts.get(e, []))
                raise SystemExit(f"[E5_DP_SERVER] oracle map {sc}: episode {e} rows {n} ≠ {v['length']}")
        self.oacts, self.osts = acts, sts
        print(
            f"[E5_DP_SERVER] ORACLE on · {len(self.omap)} scenes · clip_sample={c.clip_sample} "
            f"range={c.clip_sample_range} · log {self.olog}",
            flush=True,
        )
        if self.exact:
            print(
                f"[E5_DP_ORACLE_EXACT] on · the oracle returns the training label as is (no normalise · "
                f"clip_sample {c.clip_sample_range} · unnormalise) · roundtrip_err = the DP path's · clamp_would "
                "per query",
                flush=True,
            )

    def modality_config(self) -> dict:
        """Per modality: keys and delta indices (video and state = the n_obs latest frames)."""
        obs = list(range(-(self.n_obs - 1), 1))
        return {
            "video": {"modality_keys": list(CAMS), "delta_indices": obs},
            "state": {"modality_keys": [g for g, _, _ in GROUPS], "delta_indices": obs},
            "action": {"modality_keys": [g for g, _, _ in GROUPS], "delta_indices": list(range(self.n_act))},
            "language": {"modality_keys": [LANG], "delta_indices": [0]},
        }

    def modality_json(self) -> dict:
        """The modality file the session client verifies the server against."""
        return {
            "schema": SCHEMA,
            "embodiment_tag": "new_embodiment",
            "video": {"keys": list(CAMS), "height": self.H, "width": self.W, "horizon": self.n_obs},
            "state": {"groups": [[g, j - i] for g, i, j in GROUPS], "horizon": self.n_obs},
            "action": {"groups": [[g, j - i] for g, i, j in GROUPS], "horizon": self.n_act},
            "language": {"key": LANG},
            "source": {
                "server": "e5_dp_server",
                "checkpoint": self.ckpt,
                "root": self.root,
                "stats_sha256_12": self.stats_sha[:12],
                "n_obs_steps": self.n_obs,
                "horizon": self.horizon,
                "n_action_steps": self.n_act,
                "num_inference_steps": self.n_inf,
                **({"sampler": type(self.p.diffusion.noise_scheduler).__name__} if self.ms is not None else {}),
            },
        }

    def _last(self, x: np.ndarray) -> np.ndarray:
        """(B, T, …) → (B, n_obs, …), repeating the earliest frame if T < n_obs (LeRobot's reset behaviour)."""
        x = x[:, -self.n_obs :]
        if x.shape[1] < self.n_obs:
            x = np.concatenate([np.repeat(x[:, :1], self.n_obs - x.shape[1], axis=1), x], axis=1)
        return x

    def _axis_index(self, options, observation):
        """Task axis per batch row: options.axis_index, else looked up from each task text (unknown text raises)."""
        ax = (options or {}).get("axis_index")
        if ax is None:
            texts = observation.get("language", {}).get("human.task_description")
            if texts is None:
                raise ValueError("no language.human.task_description and no options.axis_index")
            ax = []
            for t in texts:
                t = t[0] if isinstance(t, (list, tuple)) else t
                if t not in self.text2axis:
                    raise ValueError(f"task text {t!r} not in the training tasks {list(self.text2axis)}")
                ax.append(self.text2axis[t])
        return ax

    @torch.no_grad()
    def get_action(self, observation: dict, options: dict | None = None):
        """GR00T-layout observation → ({group: (B, n_action, D)}, {}) absolute joint targets."""
        v, s = observation["video"], observation["state"]
        B = v[CAMS[0]].shape[0]
        batch = {}
        for k in CAMS:
            im = self._last(np.asarray(v[k]))  # (B, n_obs, H, W, 3) uint8
            batch[f"observation.images.{k}"] = torch.from_numpy(im).to(self.dev).permute(0, 1, 4, 2, 3).float() / 255.0
        st = np.concatenate([self._last(np.asarray(s[g], dtype=np.float32)) for g, _, _ in GROUPS], axis=-1)
        if st.shape[-1] != 56:
            raise ValueError(f"state width {st.shape[-1]} ≠ 56")
        if self.const_dims is not None and len(self.const_dims):  # constant training column → its min (normalised −1)
            stn = st.copy()
            stn[..., self.const_dims] = self.const_min
            batch["observation.state"] = torch.from_numpy(stn).to(self.dev)
        else:
            batch["observation.state"] = torch.from_numpy(st).to(self.dev)
        ax = self._axis_index(options, observation)
        ax = np.broadcast_to(np.asarray(ax, dtype=np.int64), (B,))
        env = np.zeros((B, self.n_obs, self.env_dim), np.float32)
        env[np.arange(B), :, ax] = 1.0
        batch["observation.environment_state"] = torch.from_numpy(env).to(self.dev)
        batch = self.p.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.img_keys], dim=-4)
        if self.omap is not None and self.exact:
            a = self._oracle(options or {}, B, st)  # (1, n_action, 56) the training label as is
        else:
            if self.omap is not None:
                a = self._oracle(options or {}, B, st)
            elif self.ms is None:
                a = self.p.diffusion.generate_actions(batch)  # (B, n_action, 56) normalised
            else:
                if self.dev.type == "cuda":
                    torch.cuda.synchronize(self.dev)
                t0 = time.perf_counter()
                a = self.p.diffusion.generate_actions(batch)
                if self.dev.type == "cuda":
                    torch.cuda.synchronize(self.dev)
                self.ms.append((time.perf_counter() - t0) * 1e3)
                if len(self.ms) % 50 == 0:
                    self.ms_report()
            a = self.p.unnormalize_outputs({"action": a})["action"].float().cpu().numpy()
        return {g: a[:, :, i:j] for g, i, j in GROUPS}, {}

    def _oracle(self, o: dict, B: int, st: np.ndarray):
        if B != 1 or "e5_scene" not in o or "e5_frame" not in o:
            raise ValueError(f"oracle needs B 1 and options e5_scene · e5_frame (got B {B}, {sorted(o)})")
        sc, j = str(o["e5_scene"]), int(o["e5_frame"])
        if sc not in self.omap:
            raise ValueError(f"oracle: scene {sc!r} not in the map")
        e = int(self.omap[sc]["episode"])
        A, S = self.oacts[e], self.osts[e]
        # One tail frame: the session's 30 Hz frame grid covers the 50 Hz rows by rounding up, the converter cuts by
        # rounding down, so the last query inside the session's end rule can land on j = length. That one query repeats
        # the last label (as the chunk clip inside an episode does · logged tail=1); j > length dies (unless hold_end).
        if j < 0 or (j > len(A) and not self.hold_end):
            raise ValueError(f"oracle {sc}: frame {j} outside the episode ({len(A)})")
        tail = j - len(A) + 1 if self.hold_end and j >= len(A) else j == len(A)
        j = min(j, len(A) - 1)
        idx = np.clip(np.arange(j, j + self.n_act), 0, len(A) - 1)
        serr = float(np.abs(st[0, -1] - S[j]).max())
        gt = torch.from_numpy(A[idx][None]).to(self.dev)
        a = self.p.normalize_targets({"action": gt})["action"]
        c = self.p.config
        cw = float((a.abs() - c.clip_sample_range).clamp(min=0).max()) if self.exact and c.clip_sample else 0.0
        if c.clip_sample:
            a = a.clamp(-c.clip_sample_range, c.clip_sample_range)
        rt = self.p.unnormalize_outputs({"action": a})["action"].float().cpu().numpy()[0]
        rerr = float(np.abs(rt - A[idx]).max())
        rec = {
            "t": time.time(),
            "scene": sc,
            "frame": j + int(tail),
            "state_err": serr,
            "roundtrip_err": rerr,
            "tail": int(tail),
        }
        if self.exact:
            rec["clamp_would"] = cw
        with open(self.olog, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if j == 0 and serr > 0.02:
            raise ValueError(f"oracle {sc}: frame 0 live state differs from the episode's by {serr:.4f} rad")
        if self.exact:  # the label itself — get_action skips unnormalise
            return np.ascontiguousarray(A[idx][None], dtype=np.float32)
        return a

    def ms_report(self) -> None:
        """Print the sampler's inference-time distribution so far."""
        if self.ms:
            m = np.asarray(self.ms)
            print(
                f"[E5_DP_SAMPLER] infer_ms median={np.median(m):.1f} p90={np.percentile(m, 90):.1f} "
                f"max={m.max():.1f} n={len(m)}",
                flush=True,
            )

    def reset(self, options=None):
        """Reset the policy's observation queue (one per session)."""
        self.p.reset()
        if self.const_mark:
            print(self.const_mark + " (reset)", flush=True)
        return {}
