"""dexengine.cluster: job lists, the local runner and the Slurm / Kubernetes emitters."""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

from dexengine.cluster import k8s, runner, slurm
from dexengine.cluster.jobs import Job, load_jobs, save_jobs
from dexengine.cluster.proc_ended import ENDED_STATES, ended

PY = sys.executable


def test_job_list_round_trip(tmp_path):
    jobs = [Job("a", ("echo", "1"), env={"X": "1"}, gpus=0), Job("b.2", ("echo", "2"), gpus=2, time_min=90)]
    f = save_jobs(tmp_path / "jobs.jsonl", jobs)
    assert load_jobs(f) == jobs
    with pytest.raises(ValueError):
        save_jobs(tmp_path / "dup.jsonl", [jobs[0], jobs[0]])
    (tmp_path / "bad.jsonl").write_text('{"name": "c", "cmd": ["x"], "gpu": 1}\n')
    with pytest.raises(ValueError):
        load_jobs(tmp_path / "bad.jsonl")
    for bad in ({"name": "a b", "cmd": ("x",)}, {"name": "a", "cmd": ()}, {"name": "a", "cmd": ("x",), "gpus": -1}):
        with pytest.raises(ValueError):
            Job(**bad)


def test_run_local_cpu_jobs_and_resume(tmp_path):
    out = tmp_path / "out"
    jobs = [
        Job("ok", (PY, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'] or 'none')"), gpus=0),
        Job("fail", (PY, "-c", "raise SystemExit(3)"), gpus=0),
        Job("gpu", (PY, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"), gpus=1),
    ]
    rcs = runner.run_local(jobs, ["7"], out, poll_s=0.05)
    assert rcs == {"ok": 0, "fail": 3, "gpu": 0}
    assert (out / "gpu.log").read_text().strip() == "7"
    assert (out / "fail.rc").read_text().strip() == "3"
    (out / "ok.log").unlink()
    rcs = runner.run_local(jobs, ["7"], out, poll_s=0.05)  # done jobs are skipped, failed ones rerun
    assert rcs == {"ok": 0, "fail": 3, "gpu": 0} and not (out / "ok.log").exists()
    with pytest.raises(ValueError):
        runner.run_local([Job("big", ("true",), gpus=2)], ["0"], out)


def test_slurm_script(tmp_path):
    jobs = [Job("a", ("x",), gpus=1, cpus=4, mem_gb=32, time_min=125), Job("b", ("y",), gpus=1, cpus=8)]
    text = slurm.render(
        jobs,
        "/shared/jobs.jsonl",
        "rl",
        "/shared/logs",
        slurm.SlurmConfig(partition="gpu", max_parallel=2, setup=("module load cuda",)),
    )
    for directive in (
        "--array=0-1%2",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=2:05:00",
        "--partition=gpu",
    ):
        assert f"#SBATCH {directive}" in text
    assert text.rstrip().endswith(
        '-m dexengine.cluster.job_entry --jobs /shared/jobs.jsonl --index "$SLURM_ARRAY_TASK_ID"'
    )
    assert subprocess.run(["bash", "-n"], input=text, text=True).returncode == 0


def test_k8s_manifests():
    jobs = [Job("Bundle_01", ("python", "-m", "x"), env={"A": "1"}, gpus=1, cpus=8, mem_gb=60, time_min=10)]
    cfg = k8s.K8sConfig(image="img:1", volumes={"data": {"pvc": "claim", "mount": "/data"}})
    docs = list(yaml.safe_load_all(k8s.render(jobs, cfg)))
    assert len(docs) == 1 and docs[0]["metadata"]["name"] == "dexengine-bundle-01"
    c = docs[0]["spec"]["template"]["spec"]["containers"][0]
    assert c["command"] == ["python", "-m", "x"] and c["env"] == [{"name": "A", "value": "1"}]
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert docs[0]["spec"]["activeDeadlineSeconds"] == 600
    assert docs[0]["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"] == "claim"
    with pytest.raises(ValueError):
        k8s.render(jobs)  # no image configured


def test_job_entry_runs_one_entry(tmp_path):
    f = save_jobs(
        tmp_path / "jobs.jsonl",
        [Job("a", (PY, "-c", "import os; raise SystemExit(int(os.environ['RC']))"), env={"RC": "5"}, gpus=0)],
    )
    r = subprocess.run([PY, "-m", "dexengine.cluster.job_entry", "--jobs", str(f), "--index", "0"])
    assert r.returncode == 5
    r = subprocess.run([PY, "-m", "dexengine.cluster.job_entry", "--jobs", str(f), "--index", "3"])
    assert r.returncode == 2


def test_proc_ended_states(tmp_path):
    p = subprocess.Popen([PY, "-c", "pass"])
    p.wait()
    assert set(ENDED_STATES) >= {"Z", "X"}
    assert ended(p.pid) is True


def _stamp(path, what):
    """A python -c command appending '<what> <time> <CUDA_VISIBLE_DEVICES>' to path."""
    return (
        PY,
        "-c",
        f"import os, time; open({str(path)!r}, 'a').write("
        f"'{what} %.3f %s\\n' % (time.time(), os.environ['CUDA_VISIBLE_DEVICES'] or '-'))",
    )


def test_prep_runs_before_the_gpu_is_bound(tmp_path):
    ev = tmp_path / "events.txt"
    slow = (
        PY,
        "-c",
        f"import time, os; time.sleep(0.8); open({str(ev)!r}, 'a').write("
        "'a-end %.3f %s\\n' % (time.time(), os.environ['CUDA_VISIBLE_DEVICES']))",
    )
    jobs = [
        Job("a", slow, prep=_stamp(ev, "a-prep"), gpus=1),
        Job("b", _stamp(ev, "b-cmd"), prep=_stamp(ev, "b-prep"), gpus=1),
    ]
    rcs = runner.run_local(jobs, ["7"], tmp_path / "out", poll_s=0.05)
    assert rcs == {"a": 0, "b": 0}
    rows = {w: (float(t), g) for w, t, g in (line.split() for line in ev.read_text().splitlines())}
    assert rows["a-prep"][1] == rows["b-prep"][1] == "-"  # preps see no GPU
    assert rows["a-end"][1] == rows["b-cmd"][1] == "7"
    assert rows["b-prep"][0] < rows["a-end"][0] < rows["b-cmd"][0]  # b prepares while a holds the GPU


def test_prep_failure_is_the_job_code(tmp_path):
    out, mark = tmp_path / "out", tmp_path / "ran"
    jobs = [
        Job("p", ("true",), prep=(PY, "-c", "raise SystemExit(4)"), gpus=1),
        Job("q", _stamp(mark, "q"), prep=("true",), gpus=1),
    ]
    assert runner.run_local(jobs, ["0"], out, poll_s=0.05) == {"p": 4, "q": 0}
    assert (out / "p.rc").read_text().strip() == "4" and mark.read_text().startswith("q ")


def test_prep_stored_only_when_set(tmp_path):
    a, b = Job("a", ("x",)), Job("b", ("y",), prep=("stage", "--in", "/d"))
    f = save_jobs(tmp_path / "jobs.jsonl", [a, b])
    first, second = f.read_text().splitlines()
    assert '"prep"' not in first and '"prep": ["stage", "--in", "/d"]' in second
    assert load_jobs(f) == [a, b] and load_jobs(f)[1].prep == ("stage", "--in", "/d")
    with pytest.raises(ValueError):
        Job("c", ("x",), prep=("ok", 3))


def test_job_entry_prep(tmp_path):
    mark = tmp_path / "ran"
    f = save_jobs(
        tmp_path / "jobs.jsonl",
        [
            Job("a", _stamp(mark, "cmd"), prep=_stamp(mark, "prep"), gpus=0),
            Job("b", _stamp(mark, "never"), prep=(PY, "-c", "raise SystemExit(6)")),
        ],
    )
    assert subprocess.run([PY, "-m", "dexengine.cluster.job_entry", "--jobs", str(f), "--index", "0"]).returncode == 0
    assert subprocess.run([PY, "-m", "dexengine.cluster.job_entry", "--jobs", str(f), "--index", "1"]).returncode == 6
    assert [line.split()[0] for line in mark.read_text().splitlines()] == ["prep", "cmd"]


def test_k8s_prep_init_container():
    cfg = k8s.K8sConfig(image="img:1", volumes={"data": {"pvc": "claim", "mount": "/data"}})
    job = Job("a", ("python", "-m", "x"), env={"A": "1"}, prep=("python", "-m", "stage"), gpus=1)
    pod = yaml.safe_load(k8s.render([job], cfg))["spec"]["template"]["spec"]
    init = pod["initContainers"][0]
    assert init["command"] == ["python", "-m", "stage"] and init["env"] == [{"name": "A", "value": "1"}]
    assert init["volumeMounts"] == pod["containers"][0]["volumeMounts"]
    assert "nvidia.com/gpu" not in str(init["resources"])
    assert "initContainers" not in yaml.safe_load(k8s.render([Job("b", ("y",))], cfg))["spec"]["template"]["spec"]
