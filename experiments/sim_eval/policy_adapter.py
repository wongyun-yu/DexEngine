"""E5 closed loop, policy side: chunk scheduling (row times) and the policy adapters, called by the closed loop.

Time contract (the data contract): the observation of video frame j = (image j, state row i(j)); its action =
target[i(j)+1]. Chunk element a_k (k = 0 … H−1) is the action of frame j+k = the target of row r_k = i(j+k)+1. The
target of row n = a_k of the largest k with r_k ≤ n (held until the next element — zero-order hold; interp=linear
draws a straight line between two elements). The policy is queried again every exec_horizon frames (default = the chunk
length).
Adapters:
  gr00t         policy server (experiments/sim_eval/policy_client.py, ZMQ) — observation keys and groups come from
                the server's modality JSON, the joint name order from the training set's meta (info.json names,
                modality.json slices, written by the converter); names are not written down here again.
  dp            E5 DP server (experiments/policies/dp/server.py, the gr00t wire format and job.policy form) — gr00t
  plus
                an observation history: video and state of horizon (= n_obs) consecutive frames sent as (1, T, …)
                (the closed loop observes every frame).
  replay_chunk  test adapter — returns the data episode's targets through the same chunk schedule (no server; with
                exec_horizon 1 the targets must equal the open-loop replay mode's).
"""

from __future__ import annotations

import time

from experiments.sim_eval.policy_inputs import (
    MARK,
    dataset_axis_index,
    dataset_instruction,
    dataset_layout,
    job_instruction,
    job_text_on,
    oracle_keys,
)


class Schedule:
    def __init__(self, fr: list[int], exec_horizon: int, interp: str = "hold"):
        self.fr, self.H, self.interp = fr, int(exec_horizon), interp
        self.next_q = 0
        self.rows: list[tuple[int, object]] = []  # (row, target) ascending

    def due(self, j: int) -> bool:
        return j >= self.next_q

    def put(self, j: int, chunk) -> None:
        """Chunk (T, 56) received at frame j — element k → row i(j+k)+1. Elements past the execution horizon are
        dropped
        (the next query overwrites them).
        """
        import numpy as np

        chunk = np.asarray(chunk, np.float32)
        use = min(len(chunk), self.H)
        keep = [(r, v) for r, v in self.rows if r < (self.fr[j] + 1)]
        for k in range(use):
            if j + k >= len(self.fr):
                break
            keep.append((self.fr[j + k] + 1, chunk[k]))
        self.rows = keep
        self.next_q = j + use

    def target_for_row(self, n: int):
        best, nxt = None, None
        for r, v in self.rows:
            if r <= n:
                best = (r, v)
            elif nxt is None:
                nxt = (r, v)
        if best is None:
            return None
        if self.interp == "linear" and nxt is not None:
            a = (n - best[0]) / max(1, nxt[0] - best[0])
            return (1 - a) * best[1] + a * nxt[1]
        return best[1]


#: chunk schedule defaults — the data contract's chunk of 40, execution horizon = chunk, zero-order hold. Read by the
#: policy adapters and replay_chunk.
SCHED_DEFAULT = {"chunk": 40, "exec_horizon": 40, "interp": "hold"}


#: source of the chunk values — target (default, the data contract: the action of frame j = the recorded target of row
#: i(j)+1) ·
#: next_state (test only — the action of frame j = the recorded joint state of the next frame, row i(j+1))
SOURCES = ("target", "next_state")


def parse_smooth(tag: str) -> tuple[str, int] | None:
    """Smoothing tail `c<w>` (centred moving average, w odd) · `k<w>` (causal mean of the last w, current included) →
    (kind, w). Empty = None.
    """
    if not tag:
        return None
    kind, w = tag[:1], tag[1:]
    if kind not in ("c", "k") or not w.isdigit() or int(w) < 1 or (kind == "c" and int(w) % 2 == 0):
        raise SystemExit(f"{MARK} smoothing tail {tag!r} — c<odd> or k<int>")
    return kind, int(w)


def smooth_labels(L, sm: tuple[str, int] | None):
    """Smooth the 30 Hz action label column L[j] (the recorded target of frame j); at the edges only the values present
    are averaged.
    """
    import numpy as np

    if sm is None:
        return L
    kind, w = sm
    n = len(L)
    out = np.empty_like(L)
    for j in range(n):
        a, b = (max(0, j - w // 2), min(n, j + w // 2 + 1)) if kind == "c" else (max(0, j - w + 1), j + 1)
        out[j] = L[a:b].mean(axis=0)
    return out


class ReplayChunk:
    def __init__(self, job, tg, row_to_frame, fr, params: dict | None = None):
        import numpy as np

        z = np.load(job["episode_arrays"], allow_pickle=False)
        p = dict(SCHED_DEFAULT, **(params if params is not None else (job.get("policy") or {})))
        self.source = p.get("source", "target")
        if self.source not in SOURCES:
            raise SystemExit(f"{MARK} replay_chunk source {self.source!r} — one of {SOURCES}")
        self.acts = z["action_joint_target"] if self.source == "target" else z["joint_pos"]
        self.s = Schedule(fr, int(p["exec_horizon"]), p["interp"])
        self.fr, self.chunk = fr, int(p["chunk"])
        self.smooth = parse_smooth(p.get("smooth", ""))
        # Labels of frame j (the source of element k) — target = the recorded target of row i(j)+1, next_state = the
        # recorded state of
        # row i(j+1) — are built once; smoothing (tail c<w>, k<w>) runs on this 30 Hz label column (the axis training
        # labels are smoothed on).
        last, n = len(fr) - 1, len(self.acts)
        if self.source == "target":
            idx = [min(n - 1, fr[j] + 1) for j in range(len(fr))]
        else:
            idx = [min(n - 1, fr[min(j + 1, last)]) for j in range(len(fr))]
        self.labels = smooth_labels(np.asarray(self.acts)[idx], self.smooth)

    def due(self, j):
        return self.s.due(j)

    def query(self, j, n, L, R, state):
        last = len(self.fr) - 1
        # element k = the label of frame j+k (target: recorded target of row i(j+k)+1; next_state: recorded state of row
        # i(j+k+1)), after smoothing
        self.s.put(j, self.labels[[min(j + k, last) for k in range(self.chunk)]])

    def target_for_row(self, n):
        return self.s.target_for_row(n)

    def describe(self):
        return {
            "kind": "replay_chunk",
            "exec_horizon": self.s.H,
            "interp": self.s.interp,
            "chunk": self.chunk,
            "source": self.source,
            "smooth": (f"{self.smooth[0]}{self.smooth[1]}" if self.smooth else None),
        }


class Gr00t:
    """GR00T server adapter (client = policy_client.Gr00tClient).
    job.policy = {kind: gr00t, host, port, modality: <modality JSON the server wrote after READY>, dataset_meta:
    <training
                  set folder (meta/)>, video: {<modality video key>: egoL|egoR} (complete, no default),
                  exec_horizon?, interp?, timeout_ms?, pylib?}
    Joints: scene state (tg.order) → gathered in the training set's group order (split_state); output concat_action
    (group
    order) → scene order. Group names, widths and order must agree between the server modality and the training meta
    (else death); a scene joint missing from the action names dies."""

    SCENE_ORDER = True  # chunk = scene joint order (tg.order)

    def __init__(self, job, tg, row_to_frame, fr):
        import sys

        import numpy as np

        p = job["policy"]
        for d in p.get("pylib", []):
            sys.path.insert(0, d)
        from experiments.sim_eval import policy_client as PC

        for k in ("host", "port", "modality", "dataset_meta", "video"):
            if k not in p:
                raise SystemExit(f"{MARK} job.policy.{k} missing (no default)")
        self.cli = PC.Gr00tClient(p["host"], int(p["port"]), p["modality"], timeout_ms=int(p.get("timeout_ms", 15000)))
        m = self.cli.modality
        lay = dataset_layout(p["dataset_meta"])
        for kind, mg in (("state", m.state_groups), ("action", m.action_groups)):
            got = [(g, len(n)) for g, n in lay[kind]]
            if got != [(g, int(d)) for g, d in mg]:
                raise SystemExit(f"{MARK} {kind} groups: server {mg} ≠ training set {got}")
        names = [n for _k, n in tg.order]
        pos = {n: i for i, n in enumerate(names)}
        sn = [n for _g, ns in lay["state"] for n in ns]
        an = [n for _g, ns in lay["action"] for n in ns]
        miss = [n for n in sn if n not in pos]
        if miss:
            raise SystemExit(f"{MARK} {len(miss)} policy state joints are not in the scene {miss[:4]}")
        uncovered = [n for n in names if n not in set(an)]
        if uncovered:
            raise SystemExit(f"{MARK} {len(uncovered)} scene joints are not in the policy actions {uncovered[:4]}")
        self.sidx = np.asarray([pos[n] for n in sn])  # group-order state = state[sidx]
        self.aidx = np.asarray([an.index(n) for n in names])  # scene-order action = a56[:, aidx]
        vm = dict(p["video"])
        if sorted(vm) != sorted(m.video_keys):
            raise SystemExit(
                f"{MARK} job.policy.video keys {sorted(vm)} ≠ server video keys {sorted(m.video_keys)} (all of them)"
            )
        if len(set(vm.values())) != len(vm) or set(vm.values()) - {"egoL", "egoR"}:
            raise SystemExit(f"{MARK} job.policy.video {vm} — values egoL and egoR once each")
        self.vm = vm
        self.io = {
            "frame": [],
            "row": [],
            "state": [],
            "raw": [],
            "scene": [],
        }  # server input/output record (action path check — policy_io.npz)
        self.act_names = an
        self.jt = job_text_on()
        if self.jt:  # clip-text mode (job_text_on) — off keeps the line below
            self.text, tsrc = job_instruction(job, p)
            print(f"[E5_POLICY_JOB_TEXT] on · instruction source {tsrc}", flush=True)
        else:
            self.text = job.get("instruction") or dataset_instruction(p["dataset_meta"], job["axis"])
        self.s = Schedule(
            fr, int(p.get("exec_horizon", SCHED_DEFAULT["exec_horizon"])), p.get("interp", SCHED_DEFAULT["interp"])
        )
        self.lat = []
        self.okeys = oracle_keys(job)
        print(
            f"{MARK} gr00t: server {p['host']}:{p['port']} · state groups {[(g, len(n)) for g, n in lay['state']]} · "
            f"video {vm} · horizon {m.action_horizon} execute {self.s.H} · instruction {self.text!r}",
            flush=True,
        )

    def obs(self, L, R, state) -> dict:
        import numpy as np

        o = {f"video.{k}": np.ascontiguousarray(L if v == "egoL" else R) for k, v in self.vm.items()}
        o.update(self.cli.modality.split_state(np.asarray(state, np.float32)[self.sidx]))
        o[self.cli.modality.language_key] = self.text
        return o

    def due(self, j):
        return self.s.due(j)

    def query(self, j, n, L, R, state):
        import numpy as np

        t = time.perf_counter()
        act = self.cli.get_action(self.obs(L, R, state), self.opts(j, n))
        self.lat.append(time.perf_counter() - t)
        raw = np.asarray(
            self.cli.modality.concat_action(act), np.float32
        )  # server output (group order = training action name order)
        sc = raw[:, self.aidx]
        self.io["frame"].append(j)
        self.io["row"].append(n)
        self.io["state"].append(np.asarray(state, np.float32))
        self.io["raw"].append(raw)
        self.io["scene"].append(sc)
        self.s.put(j, sc)

    def opts(self, j, n):
        return None if self.okeys is None else {**self.okeys, "e5_frame": int(j), "e5_row": int(n)}

    def save_io(self, path) -> None:
        import numpy as np

        np.savez_compressed(
            path,
            frame=np.asarray(self.io["frame"]),
            row=np.asarray(self.io["row"]),
            state=np.asarray(self.io["state"]),
            raw=np.asarray(self.io["raw"]),
            scene=np.asarray(self.io["scene"]),
            act_names=np.asarray(self.act_names),
            exec_horizon=np.int64(self.s.H),
        )

    def target_for_row(self, n):
        return self.s.target_for_row(n)

    def describe(self):
        import numpy as np

        return {
            "kind": "gr00t",
            "exec_horizon": self.s.H,
            "interp": self.s.interp,
            "instruction": self.text,
            "video": self.vm,
            "latency_s": {
                "n": len(self.lat),
                "median": float(np.median(self.lat)) if self.lat else None,
                "p90": float(np.percentile(self.lat, 90)) if self.lat else None,
            },
        }


class DP(Gr00t):
    """DP server adapter — the GR00T checks (server modality ↔ training set groups, video keys, scene joints),
    instruction
    and chunk schedule. One difference: DP (LeRobot diffusion) sees n_obs consecutive frames (training = delta
    −(n_obs−1)/30 s … 0, the episode's first frame repeated before it). The closed loop calls observe(j, …) every
    frame
    (EVERY_FRAME); a query sends the images and states of the last T (= server modality video horizon) frames as
    (1, T, …) (missing ones repeat the earliest frame, as training pads). A skipped frame dies."""

    EVERY_FRAME = True

    def __init__(self, job, tg, row_to_frame, fr):
        super().__init__(job, tg, row_to_frame, fr)
        m = self.cli.modality
        if m.video_horizon != m.state_horizon or m.video_horizon < 1:
            raise SystemExit(f"{MARK} dp: server video horizon {m.video_horizon} ≠ state horizon {m.state_horizon}")
        self.T = int(m.video_horizon)
        self.hist = []  # [(frame j, egoL, egoR, group-order state)] — the last T
        self.axi = dataset_axis_index(job["policy"]["dataset_meta"], job["axis"]) if self.jt else None
        if self.axi is not None:
            print(
                f"[E5_POLICY_JOB_TEXT] dp: options.axis_index {self.axi}({job['axis']}) — the server's "
                "text → axis table is not used",
                flush=True,
            )
        print(
            f"{MARK} dp: observation history {self.T} frames (consecutive) · action horizon "
            f"{m.action_horizon} execute {self.s.H}",
            flush=True,
        )

    def observe(self, j, n, L, R, state):
        import numpy as np

        if self.hist and j != self.hist[-1][0] + 1:
            raise SystemExit(f"{MARK} dp: frame {j} does not follow the last observation {self.hist[-1][0]}")
        st = np.asarray(state, np.float32)[self.sidx]
        self.hist.append((j, np.ascontiguousarray(L), np.ascontiguousarray(R), st))
        del self.hist[: -self.T]

    def query(self, j, n, L, R, state):
        import numpy as np

        if not self.hist or self.hist[-1][0] != j:
            raise SystemExit(f"{MARK} dp: no observe before the query at frame {j}")
        m = self.cli.modality
        H, W = m.image_hw
        h = [self.hist[0]] * (self.T - len(self.hist)) + self.hist
        video = {}
        for k, v in self.vm.items():
            a = np.stack([x[1] if v == "egoL" else x[2] for x in h])[None]
            if a.dtype != np.uint8 or a.shape != (1, self.T, H, W, 3):
                raise SystemExit(f"{MARK} dp: video.{k} {a.dtype} {a.shape} ≠ uint8 (1, {self.T}, {H}, {W}, 3)")
            video[k] = np.ascontiguousarray(a)
        st = np.stack([x[3] for x in h])
        if st.shape != (self.T, m.state_dim) or not np.all(np.isfinite(st)):
            raise SystemExit(f"{MARK} dp: state {st.shape} ≠ ({self.T}, {m.state_dim}) or not finite")
        state_g, i = {}, 0
        for g, d in m.state_groups:
            state_g[g] = np.ascontiguousarray(st[None, :, i : i + d])
            i += d
        req = {"video": video, "state": state_g, "language": {m.language_key: [[self.text]]}}
        t = time.perf_counter()
        act = {k: v[0] for k, v in self.cli._request(req, 1, self.opts(j, n)).items()}
        self.lat.append(time.perf_counter() - t)
        raw = np.asarray(m.concat_action(act), np.float32)
        sc = raw[:, self.aidx]
        for k, v in (("frame", j), ("row", n), ("state", np.asarray(state, np.float32)), ("raw", raw), ("scene", sc)):
            self.io[k].append(v)
        self.s.put(j, sc)

    def opts(self, j, n):
        o = super().opts(j, n)
        if self.axi is None:  # off = the old value
            return o
        return {**(o or {}), "axis_index": int(self.axi)}

    def describe(self):
        d = super().describe()
        d.update(kind="dp", obs_frames=self.T)
        return d


KINDS = {"replay_chunk": ReplayChunk, "gr00t": Gr00t, "dp": DP}


def make(job, tg, row_to_frame, fr):
    k = (job.get("policy") or {}).get("kind")
    if k not in KINDS:
        raise SystemExit(f"{MARK} policy kind {k!r} — one of {sorted(KINDS)} (no default)")
    return KINDS[k](job, tg, row_to_frame, fr)
