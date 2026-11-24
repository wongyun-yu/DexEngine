"""Training stats = inference stats, for a GR00T checkpoint (every policy server checks this at start and dies on a
difference). Three things must be the same JSON (sha256 of the canonical dump, sorted keys):
  train  = <ckpt>/experiment_cfg/dataset_statistics.json[tag]  (written once at training start from the train dataset —
           gr00t/model/base/model_pipeline.py) — what training normalised with
  saved  = <ckpt>/statistics.json[tag]                          (processor.save_pretrained at every checkpoint)
  loaded = the processor Gr00tPolicy builds from <ckpt> (AutoProcessor.from_pretrained → state_action_processor) —
           what the server / open-loop eval normalise with (--load-processor; needs the Isaac-GR00T venv)
Prints [E5_STATS_CHECK] gr00t … and returns the sha; check() raises SystemExit on any difference.
With --load-processor also the normalisation gate (the eval side must prove what it normalises with):
[E5_NORM] use_percentiles=<built processor> state_action=<its state_action_processor>
train_cfg=<experiment_cfg/conf.yaml model.use_percentiles — the training intent>
train_proc=<experiment_cfg/final_processor_config.json use_percentiles —
the processor training used> — any two differ → SystemExit. Then one round trip through the built processor
(apply_action → unapply_action) of a label at each action dim's training max (the value percentile clipping cuts to
q99): [E5_NORM] roundtrip max_err=… (percentile cut would be …) — with use_percentiles False an error > 1e-4 rad is
a SystemExit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canon_sha(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check(ckpt: str, tag: str = "new_embodiment", processor=None) -> str:
    c = Path(ckpt)
    pdir = c if (c / "processor_config.json").exists() else c / "processor"
    train = json.loads(Path(c / "experiment_cfg" / "dataset_statistics.json").read_text())[tag]
    saved = json.loads(Path(pdir / "statistics.json").read_text())[tag]
    shas = {"train": canon_sha(train), "saved": canon_sha(saved)}
    if processor is not None:
        from gr00t.data.utils import to_json_serializable

        shas["loaded"] = canon_sha(
            json.loads(json.dumps(to_json_serializable(processor.state_action_processor.statistics[tag])))
        )
    same = len(set(shas.values())) == 1
    print(
        f"[E5_STATS_CHECK] gr00t {'OK' if same else 'FAIL'} tag={tag} "
        + " ".join(f"{k}={v[:12]}" for k, v in shas.items())
        + f" ckpt={ckpt}",
        flush=True,
    )
    if not same:
        raise SystemExit(f"training stats ≠ inference stats for {ckpt}: {shas}")
    return shas["train"]


def norm_check(ckpt: str, proc, tag: str = "new_embodiment") -> None:
    import re

    import numpy as np

    c = Path(ckpt)
    txt = (c / "experiment_cfg" / "conf.yaml").read_text()
    m = re.findall(r"^\s*use_percentiles:\s*(\w+)", txt, re.M)
    if len(m) != 1:
        raise SystemExit(f"[E5_NORM] {ckpt}: conf.yaml has {len(m)} use_percentiles lines (want 1)")
    cfg = m[0].lower() == "true"
    with open(c / "experiment_cfg" / "final_processor_config.json") as fh:
        fp = json.load(fh).get("use_percentiles")
    if str(fp) not in ("True", "False"):
        raise SystemExit(f"[E5_NORM] {ckpt}: final_processor_config.json use_percentiles {fp!r}")
    sap = proc.state_action_processor
    got = (bool(proc.use_percentiles), bool(sap.use_percentiles), cfg, str(fp) == "True")
    print(
        f"[E5_NORM] use_percentiles={got[0]} state_action={got[1]} train_cfg={got[2]} train_proc={got[3]} ckpt={ckpt}",
        flush=True,
    )
    if len(set(got)) != 1:
        raise SystemExit(
            f"[E5_NORM] {ckpt}: processor {got[0]} · state_action {got[1]} · training config {got[2]} · "
            f"training processor {got[3]} differ"
        )
    st = sap.statistics[tag]
    state = {g: np.asarray(v["mean"], np.float32)[None] for g, v in st.get("state", {}).items()}
    act = {g: np.asarray(v["max"], np.float32)[None] for g, v in st["action"].items()}
    cut = max(float(np.max(np.asarray(v["max"]) - np.asarray(v["q99"]))) for v in st["action"].values())
    rt = sap.unapply_action(sap.apply_action(act, tag, state), tag, state)
    err = max(float(np.max(np.abs(np.asarray(rt[g]).reshape(-1) - act[g].reshape(-1)))) for g in act)
    print(
        f"[E5_NORM] roundtrip max_err={err:.2e} rad at each action dim's training max "
        f"(percentile cut would be up to {cut:.3f})",
        flush=True,
    )
    if not got[0] and err > 1e-4:
        raise SystemExit(f"[E5_NORM] {ckpt}: min/max processor still cuts the training max by {err:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="new_embodiment")
    ap.add_argument("--load-processor", action="store_true")
    a = ap.parse_args()
    proc = None
    if a.load_processor:
        import gr00t.model  # noqa: F401 — registers the GR00T processor classes
        from transformers import AutoProcessor

        c = Path(a.ckpt)
        proc = AutoProcessor.from_pretrained(c if (c / "processor_config.json").exists() else c / "processor")
    check(a.ckpt, a.tag, proc)
    if proc is not None:
        norm_check(a.ckpt, proc, a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
