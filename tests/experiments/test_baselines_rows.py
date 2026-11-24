"""Tab. 2 baseline row builders: the ManipTrans aggregateMode override and the SPIDER (a) row swaps."""

import pytest

from experiments.baselines.common import method_cmds, prep_mass_0p2
from experiments.baselines.maniptrans import agg_override as ao

CLIP = "brush_brush_cup__20231026_012"  # one of agg_override.ROWS
MT_LINE = (
    "cd /w/code/maniptrans && export PATH=/e/bin:$PATH && budget 7200 /e/bin/python main/rl/train.py task=ResDexHand "
    "dexhand=sharpa_wave seed=42 > /w/logs/x.log 2>&1; timeout 1800 env PYTHONPATH=/r /e/bin/python "
    "/r/experiments/baselines/maniptrans/eval_record.py /w/results/maniptrans/x.npz -- task=ResDexHand "
    "dexhand=sharpa_wave "
    "num_envs=64 > /w/logs/x.eval.log 2>&1"
)


def test_agg_override_both_calls_once_and_idempotent():
    out = ao.rewrite(MT_LINE, CLIP)
    assert out.startswith(ao.marker(CLIP)) and out.count(ao.OVR) == 2
    assert "main/rl/train.py task=ResDexHand task.env.aggregateMode=0 " in out
    assert " -- task=ResDexHand task.env.aggregateMode=0 dexhand" in out
    assert ao.rewrite(out, CLIP) == out
    with pytest.raises(SystemExit):
        ao.rewrite(MT_LINE.replace("main/rl/train.py task=ResDexHand", "main/rl/train.py task=X"), CLIP)
    assert method_cmds.mt_agg(MT_LINE, "some_other_clip") == MT_LINE  # only the listed clips get it
    assert method_cmds.mt_agg(MT_LINE, CLIP) == out


def test_object_mass_owner_value():
    assert prep_mass_0p2.owner_mass() == 0.2
    assert method_cmds.owner_mass() == 0.2


def test_spider_a_nocheck_m0p2_row(tmp_path, monkeypatch):
    assert method_cmds.sub("a b a", "a", "c", 2) == "c b c"
    with pytest.raises(SystemExit):  # a swap that hits a different count stops the build
        method_cmds.sub("a b a", "a", "c", 1)
    B = str(tmp_path)
    (tmp_path / "run").mkdir()
    c = "clip_a"
    line = (
        f"cd {B}/code/spider && CUDA_VISIBLE_DEVICES=${{GPU:-0}} timeout 7200 .venv/bin/python examples/run_mjwp.py "
        f"+override=taco_sharpa_origin task={c} dataset_dir={B}/data/spider > {B}/logs/spider/{c}.log 2>&1; "
        f"echo rc=$? >> {B}/logs/spider/{c}.log; env -u LD_LIBRARY_PATH /fc/python -I /r/score.py spider "
        f"--clip-dir {B}/data/spider/processed/taco/sharpa_wave/bimanual/{c} --sim-dt 0.01 --out "
        f"{B}/results/spider/{c}.json"
    )
    (tmp_path / "run/spider_cmds.txt").write_text(line + "\n")
    monkeypatch.setenv("DEXENGINE_E123_ROOT", B)
    cmd = method_cmds.spider_a_cmd(c, "off", mass="m0p2")
    assert f"dataset_dir={B}/data/spider_a_m0p2 +sanity_check_seconds=0.0 " in cmd
    assert (
        f"-I {method_cmds.MASS_SETS['m0p2']} --check {B}/data/spider_a_m0p2 {c} > "
        f"{B}/logs/spider/{c}.start_a_nocheck_m0p2.log" in cmd
    )
    assert f"--out {B}/results/spider/{c}.start_a_nocheck_m0p2.json" in cmd
    assert "[SPIDER_STARTCHECK] off rc=$rc applied" in cmd and f"{B}/data/spider/" not in cmd
