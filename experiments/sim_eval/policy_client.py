#!/usr/bin/env python3
"""E5 GR00T N1.7 policy client — wire-compatible with Isaac-GR00T ``n1.7-release`` (23ace64f) ``PolicyServer``.

Importable from the Isaac interpreter: it needs only numpy + pyzmq + msgpack (a ``pip --target`` folder put on
sys.path through the policy JSON's ``pylib`` list; the Isaac env is never modified)::

    from experiments.sim_eval.policy_client import Gr00tClient
    cli = Gr00tClient("127.0.0.1", 5555, "<modality>.json")
    act = cli.get_action(obs)          # {"left_arm_joints": (40, 7) float32, ...}

Wire format = ``gr00t/policy/server_client.py`` @ n1.7-release, copied exactly:
  * msgpack ``packb(default=encode)`` / ``unpackb(object_hook=decode)``                    (L33, L37)
  * ndarray -> ``{"__ndarray_class__": True, "as_npy": np.save(..., allow_pickle=False)}``  (L56-59)
  * decode ndarray with ``np.load(BytesIO, allow_pickle=False)``                            (L45-46)
  * ModalityConfig arrives as ``{"__ModalityConfig_class__": True, "as_json": {...}}``      (L51-55) -> plain dict here
  * request ``{"endpoint": e, "data": {...}}`` — no ``data`` key when the endpoint takes none (L221-223)
  * ``get_action`` data = ``{"observation": obs, "options": options}``, reply = ``[action, info]`` (L252-255)
  * reply dict with ``"error"`` -> server-side exception (L159-164, L240-241)
  * timeout = ``zmq.error.Again``; the REQ socket is then unusable, so it is rebuilt and the error re-raised (L227-235)

Observation layout the server checks (``gr00t/policy/gr00t_policy.py`` check_observation, L208-360):
  video uint8 (B, T, H, W, 3) · state float32 (B, T, D) · language list[list[str]] (B x T=1).
This client takes ONE frame per key (B = 1) under flat contract keys::

    "video.<cam>": uint8 (H, W, 3) or (1, H, W, 3)
    "state.<group>": float32 (D,) or (1, D)
    "<language key>": str          (e.g. "annotation.human.task_description")

and adds the batch/time axes itself. Group names, dims, cameras, image size and horizons are NOT hard-coded:
they come from a modality JSON (schema ``e5_policy_modality/1``, written by ``make-modality`` below from the
checkpoint and/or the dataset meta), and are cross-checked against the server's ``get_modality_config`` at
connect time. Any mismatch raises — there is no fallback path.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from experiments.sim_eval.policy_modality import Modality, ModalityMismatch, modality_from_sources

try:
    import msgpack
    import zmq
except ImportError as e:  # loud: the Isaac interpreter needs the pip --target dir on sys.path
    raise ImportError(
        f"{e}. policy_client needs pyzmq + msgpack; put their pip --target folder on sys.path "
        "(the policy JSON's pylib list)."
    ) from e


class PolicyTimeout(TimeoutError):
    """The server did not answer within timeout_ms (socket was rebuilt; the request is lost)."""


class PolicyServerError(RuntimeError):
    """The server answered with an error dict (exception inside the policy)."""


# ----------------------------------------------------------------------------------------------- wire format
def _encode_custom(obj: Any) -> Any:
    # server_client.py L50-60 (ModalityConfig branch omitted: the client never sends one)
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode_custom(obj: Any) -> Any:
    # server_client.py L40-47; ModalityConfig is returned as its JSON dict (no gr00t import on the Isaac side)
    if not isinstance(obj, dict):
        return obj
    if "__ModalityConfig_class__" in obj:
        return obj["as_json"]
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def to_bytes(data: Any) -> bytes:
    return msgpack.packb(data, default=_encode_custom)


def from_bytes(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode_custom)


# ----------------------------------------------------------------------------------------------- client
class Connection:
    """One REQ socket to a GR00T PolicyServer (server_client.py PolicyClient L172-242, minus the silent ping)."""

    def __init__(self, host: str, port: int, timeout_ms: int = 15000, api_token: str | None = None):
        self.host, self.port, self.timeout_ms, self.api_token = host, int(port), int(timeout_ms), api_token
        self._ctx = zmq.Context()
        self._sock = None
        self._init_socket()

    def _init_socket(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        s = self._ctx.socket(zmq.REQ)
        s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        s.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(f"tcp://{self.host}:{self.port}")
        self._sock = s

    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        req: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        if self.api_token:
            req["api_token"] = self.api_token
        try:
            self._sock.send(to_bytes(req))
            msg = self._sock.recv()
        except zmq.error.Again as e:
            self._init_socket()  # REQ is stuck waiting for a reply that will never come
            raise PolicyTimeout(
                f"{endpoint}: no reply from tcp://{self.host}:{self.port} in {self.timeout_ms} ms"
            ) from e
        if msg == b"ERROR":
            raise PolicyServerError("server replied b'ERROR' — wrong server on this port?")
        resp = from_bytes(msg)
        if isinstance(resp, dict) and "error" in resp:
            raise PolicyServerError(f"{endpoint}: {resp['error']}")
        return resp

    def ping(self) -> bool:
        """True, or raises PolicyTimeout / PolicyServerError (GR00T's own ping returns False instead)."""
        resp = self.call("ping", requires_input=False)
        if not (isinstance(resp, dict) and resp.get("status") == "ok"):
            raise PolicyServerError(f"unexpected ping reply {resp!r}")
        return True

    def get_modality_config(self) -> dict:
        return self.call("get_modality_config", requires_input=False)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        self._ctx.term()


class Gr00tClient(Connection):
    """REQ client for one GR00T PolicyServer, bound to a modality JSON. Not thread-safe (one REQ socket)."""

    def __init__(
        self,
        host: str,
        port: int,
        modality: Modality | dict | str | os.PathLike,
        timeout_ms: int = 15000,
        api_token: str | None = None,
        verify: bool = True,
    ):
        self.modality = Modality.load(modality)
        super().__init__(host, port, timeout_ms=timeout_ms, api_token=api_token)
        self.server_cfg: dict | None = None
        self.last_info: Any = None
        if verify:
            self.ping()
            self.verify_server()

    def verify_server(self) -> dict:
        """Server (= checkpoint) keys and horizons must equal the modality JSON's."""
        cfg = self.get_modality_config()
        if not cfg:
            raise ModalityMismatch("server returned an empty modality config (not a Gr00tPolicy server?)")
        m = self.modality
        want = {
            ("video", "modality_keys"): m.video_keys,
            ("state", "modality_keys"): [k for k, _ in m.state_groups],
            ("action", "modality_keys"): [k for k, _ in m.action_groups],
            ("language", "modality_keys"): [m.language_key],
        }
        for (mod, f), v in want.items():
            if list(cfg[mod][f]) != list(v):
                raise ModalityMismatch(f"server {mod}.{f} {cfg[mod][f]} != modality JSON {v}")
        for mod, h in (("video", m.video_horizon), ("state", m.state_horizon), ("action", m.action_horizon)):
            if len(cfg[mod]["delta_indices"]) != h:
                raise ModalityMismatch(f"server {mod} horizon {len(cfg[mod]['delta_indices'])} != modality JSON {h}")
        self.server_cfg = cfg
        return cfg

    def reset(self, options: dict | None = None) -> Any:
        return self.call("reset", {"options": options})

    def build_request_obs(self, obs: dict, batch: int | None = None) -> dict:
        """Flat contract keys -> GR00T nested batch (B, T=1). Strict: unknown or missing keys raise.

        batch=None: one env — video (H,W,3)|(1,H,W,3), state (d,)|(1,d), language str.
        batch=B:    B envs — video (B,H,W,3), state (B,d), language str (shared) or list of B str."""
        m = self.modality
        H, W = m.image_hw
        want = {f"video.{k}" for k in m.video_keys} | {f"state.{k}" for k, _ in m.state_groups} | {m.language_key}
        if set(obs) != want:
            raise ModalityMismatch(
                f"observation keys: missing {sorted(want - set(obs))}, unexpected {sorted(set(obs) - want)}"
            )
        B = 1 if batch is None else int(batch)
        video = {}
        for k in m.video_keys:
            a = np.asarray(obs[f"video.{k}"])
            if batch is None and a.ndim == 3:
                a = a[None]
            if a.dtype != np.uint8 or a.shape != (B, H, W, 3):
                raise ModalityMismatch(f"video.{k}: want uint8 ({B},{H},{W},3), got {a.dtype} {a.shape}")
            video[k] = np.ascontiguousarray(a[:, None])  # (B, T=1, H, W, 3)
        state = {}
        for k, d in m.state_groups:
            a = np.asarray(obs[f"state.{k}"])
            if batch is None and a.ndim == 1:
                a = a[None]
            if a.dtype != np.float32 or a.shape != (B, d):
                raise ModalityMismatch(f"state.{k}: want float32 ({B},{d}), got {a.dtype} {a.shape}")
            if not np.all(np.isfinite(a)):
                raise ModalityMismatch(f"state.{k} has non-finite values")
            state[k] = np.ascontiguousarray(a[:, None])  # (B, T=1, d)
        text = obs[m.language_key]
        texts = [text] * B if isinstance(text, str) else list(text)
        if len(texts) != B or not all(isinstance(t, str) and t for t in texts):
            raise ModalityMismatch(f"{m.language_key}: want a non-empty str or {B} non-empty str")
        return {"video": video, "state": state, "language": {m.language_key: [[t] for t in texts]}}

    def get_action(self, obs: dict, options: dict | None = None) -> dict[str, np.ndarray]:
        """One env -> {"<action group>": float32 (T=action_horizon, d)}; raises on any shape/key/NaN mismatch."""
        out = self._request(self.build_request_obs(obs), 1, options)
        return {k: v[0] for k, v in out.items()}

    def _request(self, req: dict, B: int, options: dict | None) -> dict[str, np.ndarray]:
        resp = self.call("get_action", {"observation": req, "options": options})
        if not (isinstance(resp, (list, tuple)) and len(resp) == 2 and isinstance(resp[0], dict)):
            raise PolicyServerError(f"get_action reply is not [action, info]: {type(resp).__name__}")
        action, self.last_info = resp
        m = self.modality
        want = [k for k, _ in m.action_groups]
        if sorted(action) != sorted(want):
            raise ModalityMismatch(f"server action keys {sorted(action)} != modality {want}")
        out = {}
        for k, d in m.action_groups:
            a = np.asarray(action[k])
            if a.shape != (B, m.action_horizon, d):
                raise ModalityMismatch(f"action {k}: want ({B},{m.action_horizon},{d}), got {a.shape}")
            if not np.all(np.isfinite(a)):
                raise ModalityMismatch(f"action {k} has non-finite values")
            out[k] = a.astype(np.float32, copy=False)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ----------------------------------------------------------------------------------------------- CLI
def cmd_make_modality(a: argparse.Namespace) -> int:
    server_cfg = None
    if a.host:
        conn = Connection(a.host, a.port, timeout_ms=a.timeout_ms)
        conn.ping()
        server_cfg = conn.get_modality_config()
        conn.close()
    hw = tuple(int(x) for x in a.image_hw.lower().split("x")) if a.image_hw else None
    m = modality_from_sources(
        checkpoint=a.checkpoint, dataset=a.dataset, server_cfg=server_cfg, embodiment_tag=a.embodiment_tag, image_hw=hw
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(m.to_json(), indent=1))
    print(
        f"[E5CLIENT] modality -> {a.out}: video {m.video_keys} {m.image_hw} · state {m.state_groups} "
        f"(D={m.state_dim}) · action D={m.action_dim} H={m.action_horizon} · lang {m.language_key}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mm = sub.add_parser("make-modality", help="write the modality JSON from checkpoint / dataset / live server")
    mm.add_argument("--checkpoint", default=None)
    mm.add_argument("--dataset", default=None, help="LeRobot dir with meta/modality.json + meta/info.json")
    mm.add_argument("--host", default=None, help="also cross-check against a live server")
    mm.add_argument("--port", type=int, default=5555)
    mm.add_argument("--timeout-ms", type=int, default=15000)
    mm.add_argument("--embodiment-tag", default="new_embodiment")
    mm.add_argument("--image-hw", default=None, help="HxW when no dataset is given, e.g. 180x320")
    mm.add_argument("--out", required=True)
    mm.set_defaults(fn=cmd_make_modality)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
