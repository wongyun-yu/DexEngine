"""E5 closed loop, policy inputs: what the policy adapters read besides the server — the training set meta
(joint name order, group slices, instruction, axis index) and the session options that put the clip text or the
oracle keys into each query. Called by experiments/sim_eval/policy_adapter.py.
"""

from __future__ import annotations

import json
from pathlib import Path

MARK = "[E5_POLICY]"


def dataset_layout(meta_dir: str | Path) -> dict:
    """Training set meta (info.json joint names + modality.json group slices) → {"state"|"action": [(group, [joint
    names…]), …]}. The joint name order lives in the training set (the converter's meta); it is not written down here.
    Dies on repeated names, a slice beyond the name count, and gaps or overlaps between slices."""
    meta = Path(meta_dir)
    if (meta / "meta").is_dir():
        meta = meta / "meta"
    info = json.loads((meta / "info.json").read_text())
    mj = json.loads((meta / "modality.json").read_text())
    feat = {"state": "observation.state", "action": "action"}
    out = {}
    for m, fk in feat.items():
        names = list(info["features"][fk]["names"])
        if len(set(names)) != len(names):
            raise SystemExit(f"{MARK} {meta}/info.json {fk}: repeated names")
        spans = sorted(mj[m].items(), key=lambda kv: int(kv[1]["start"]))
        cur, groups = 0, []
        for g, sp in spans:
            s0, s1 = int(sp["start"]), int(sp["end"])
            if s0 != cur or s1 <= s0 or s1 > len(names):
                raise SystemExit(
                    f"{MARK} {meta}/modality.json {m}.{g} slice [{s0},{s1}) — "
                    f"not contiguous or beyond {len(names)} names"
                )
            groups.append((g, names[s0:s1]))
            cur = s1
        if cur != len(names):
            raise SystemExit(f"{MARK} {meta}/modality.json {m} slice sum {cur} ≠ names {len(names)}")
        out[m] = groups
    return out


def dataset_instruction(meta_dir: str | Path, axis: str) -> str:
    """Instruction = the text of the training set's episodes of this task axis (e5_axis) — there must be exactly one.
    With
    several (a text per clip) job.instruction must be given; no choice is made here."""
    meta = Path(meta_dir)
    if (meta / "meta").is_dir():
        meta = meta / "meta"
    texts = set()
    with open(meta / "episodes.jsonl") as fh:
        for ln in fh:
            e = json.loads(ln)
            if e.get("e5_axis") == axis:
                texts.update(e["tasks"])
    if len(texts) != 1:
        raise SystemExit(f"{MARK} training set axis {axis} has {len(texts)} instructions — give job.instruction")
    return texts.pop()


def job_text_on() -> bool:
    """Session option policy_job_text=1 (default off): clip-text training sets (several texts per task axis). On:
    (1) instruction = job.instruction or the policy's instructions[scene] (scene = the folder name of job.plan; the
    policy
    JSON carries the map {scene: text} or a path to it) — neither dies (no fallback to the training set's text);
    (2) DP sends options.axis_index per query = the position of job.axis in the training set's conversion record
    (meta/e5_conversion.json axes) instead of the server's text → axis table (an unseen scene text is no training
    text).
    Marker [E5_POLICY_JOB_TEXT]."""
    from experiments.sim_eval.session_cfg import value

    v = value("policy_job_text", "")
    if v in ("", "0"):
        return False
    if v != "1":
        raise SystemExit(f"{MARK} session option policy_job_text={v!r} — 0 or 1")
    return True


def job_instruction(job, p) -> tuple[str, str]:
    """The instruction when on, and where it came from (job_text_on)."""
    t = (job.get("instruction") or "").strip()
    if t:
        return t, "job.instruction"
    scene = Path(job["plan"]).parent.name if job.get("plan") else None
    ins = p.get("instructions")
    if isinstance(ins, str):
        ins = json.loads(Path(ins).read_text())
    t = ((ins or {}).get(scene) or "").strip() if scene else ""
    if not t:
        raise SystemExit(
            f"{MARK} policy_job_text=1 — no instruction for scene {scene!r} (neither job.instruction nor "
            f"policy.instructions[scene]; no default)"
        )
    return t, f"policy.instructions[{scene}]"


def dataset_axis_index(meta_dir: str | Path, axis: str) -> int:
    """DP axis index = the axes of the training set's meta/e5_conversion.json (the converter's AXES order, the table of
    the
    training frames' axis_index)."""
    meta = Path(meta_dir)
    if (meta / "meta").is_dir():
        meta = meta / "meta"
    axes = json.loads((meta / "e5_conversion.json").read_text()).get("axes")
    if not axes or axis not in axes:
        raise SystemExit(f"{MARK} policy_job_text=1 — job.axis {axis!r} is not in the training set axes {axes}")
    return axes.index(axis)


def oracle_keys(job) -> dict | None:
    """Session option policy_oracle_keys=1 (default off — options None): every query carries the scene name, frame and
    row
    in options. Only oracle servers (ground-truth chunks) read them; the real GR00T server ignores options
    (_get_action 'currently unused') and the DP server reads only axis_index."""
    from experiments.sim_eval.session_cfg import value

    v = value("policy_oracle_keys", "")
    if v in ("", "0"):
        return None
    if v != "1":
        raise SystemExit(f"{MARK} session option policy_oracle_keys={v!r} — 0 or 1")
    if not job.get("plan") or not job.get("clip"):
        raise SystemExit(f"{MARK} policy_oracle_keys=1 but the job has no plan or clip")
    k = {"e5_scene": Path(job["plan"]).parent.name, "e5_clip": job["clip"]}
    print(f"[E5_POLICY_ORACLE_KEYS] on · {k}", flush=True)
    return k
