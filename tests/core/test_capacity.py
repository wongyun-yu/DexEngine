"""capacity is the one owner of pool sizes: it reads the machine (cgroup cpu.max · memory.max, else the host), sizes a
pool from the step's memory statement, preloads the native modules a thread pool's jobs import, and every pool in
dexengine goes through it (structural check below, with planted violations)."""

from __future__ import annotations

import ast
import csv
import multiprocessing as mp
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from dexengine.core import capacity as C
from dexengine.core import machine as M
from dexengine.core.repo import REPO_ROOT

POOLS = {"Pool", "ProcessPoolExecutor", "ThreadPoolExecutor"}


def _cg(tmp_path, cpu="max 100000", mem="max", parent_mem="max"):
    root = tmp_path / "cg"
    leaf = root / "a.slice" / "b.scope"
    leaf.mkdir(parents=True)
    (leaf / "cpu.max").write_text(cpu + "\n")
    (leaf / "memory.max").write_text(mem + "\n")
    (leaf.parent / "memory.max").write_text(parent_mem + "\n")
    proc = tmp_path / "cgroup"
    proc.write_text("0::/a.slice/b.scope\n")
    return C.cgroup_dirs(str(proc), root)


def test_the_machine_is_read_from_the_cgroup_chain(tmp_path, monkeypatch):
    dirs = _cg(tmp_path, cpu="150000 100000", mem=str(3 * 2**30), parent_mem=str(2 * 2**30))
    monkeypatch.setattr(C.os, "sched_getaffinity", lambda _p: set(range(24)))
    assert C.cpu_limit(dirs) == (2, "cpu.max")  # 1.5 CPUs round up (module doc: batch throughput)
    monkeypatch.setattr(M, "_host_available", lambda meminfo="": 100 * 2**30)
    assert C.mem_limit(dirs) == (2 * 2**30, "memory.max")  # the smallest of the chain (the parent here)
    dirs = _cg(tmp_path / "b")
    assert C.cpu_limit(dirs) == (24, "affinity")
    assert C.mem_limit(dirs) == (100 * 2**30, "host")


def _plan(monkeypatch, need, jobs=100, option=None, share=None, cpus=24, gib=20.0):
    monkeypatch.setitem(C.NEEDS, "t.step", need)
    monkeypatch.setattr(C, "cpu_limit", lambda dirs=None: (cpus, "affinity"))
    monkeypatch.setattr(C, "mem_limit", lambda dirs=None: (int(gib * 2**30), "memory.max"))
    return C.plan("t.step", jobs, option, share=share)


def test_the_plan_takes_the_smallest_of_cpus_memory_jobs_and_cap(monkeypatch):
    need = C.Need(0.5, 4.2, 8, measured="r")
    assert _plan(monkeypatch, need).workers == 3  # (20 x 0.85 - 0.5) // 4.2 (the 20 GiB canary scope)
    assert _plan(monkeypatch, need, gib=8.0).workers == 1  # a small machine
    assert _plan(monkeypatch, need, gib=500.0).workers == 8  # a large one runs as before (the old fixed count)
    assert _plan(monkeypatch, need, cpus=2, gib=500.0).workers == 2
    assert _plan(monkeypatch, need, jobs=1, gib=500.0).workers == 1
    assert _plan(monkeypatch, need, share=5, gib=500.0).workers == 5  # a lane's CPU share caps the CPUs
    p = _plan(monkeypatch, need, option=7, gib=8.0)  # an explicit option replaces the owner's value, shown with it
    assert (p.workers, p.source, p.owner_workers) == (7, "option", 1)
    assert "owner_workers=1" in p.line()
    assert _plan(monkeypatch, C.Need(1, 2, 16, kind="thread"), cpus=4, gib=1.0).workers == 4  # memory: not a bound
    assert _plan(monkeypatch, C.Need(1, 2, 16, kind="io"), cpus=1, gib=1.0).workers == 16  # waits: neither
    with pytest.raises(SystemExit, match="no memory statement"):
        C.plan("t.unknown", 4)
    with pytest.raises(SystemExit, match="< 1"):
        _plan(monkeypatch, need, option=0)


def test_threads_are_one_unless_the_caller_set_them(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert C.threads() == (1, "owner")
    env = C.child_env(1, {"OMP_NUM_THREADS": "4"})
    assert env["OMP_NUM_THREADS"] == "4" and env["PXR_WORK_THREAD_LIMIT"] == "1"  # caller kept · USD TBB capped
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert C.threads() == (3, "env")


def _records() -> list[dict]:
    rows = (Path(C.__file__).with_name("capacity_peaks.tsv")).read_text(encoding="utf-8").splitlines()
    return list(csv.DictReader([r for r in rows if not r.startswith("#")], delimiter="\t"))


def test_every_measured_statement_covers_its_records():
    recs = _records()
    named = {(r["step"], r["record"]) for r in recs}
    for step, need in C.NEEDS.items():
        if not need.measured:
            continue
        mine = [r for r in recs if r["step"] == step and r["record"] == need.measured]
        assert mine, f"{step}: statement names record {need.measured!r}, which capacity_peaks.tsv does not hold"
        for r in mine:  # statement >= MARGIN x every record (rounded to the 0.1 GiB a statement is written in)
            assert need.per_job_gib >= round(C.MARGIN * float(r["per_job_gib"]), 1), (step, r)
            if r["parent_gib"] != "-":
                assert need.parent_gib >= round(C.MARGIN * float(r["parent_gib"]), 1), (step, r)
    stray = {s for s, rec in named if C.NEEDS.get(s, C.Need(0, 0, 1)).measured != rec}
    assert not stray, f"records no statement names: {stray}"


def _imports(tree: ast.AST) -> set[str]:
    """Names in this module that are the pool classes or the pool modules."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in ("multiprocessing", "concurrent"):
            out |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Import):
            out |= {a.asname or a.name for a in n.names if a.name.split(".")[0] in ("multiprocessing", "concurrent")}
    return out


def _pool_call(call: ast.Call, names: set[str]) -> bool:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id in POOLS and f.id in names
    if isinstance(f, ast.Attribute) and f.attr in POOLS:
        base = f.value
        while isinstance(base, (ast.Attribute, ast.Call)):
            base = base.func if isinstance(base, ast.Call) else base.value
        return isinstance(base, ast.Name) and base.id in names
    return False


def _callee(call: ast.Call) -> str:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def _initialized(call: ast.Call) -> bool:
    """A process pool starts its workers with capacity.worker_init (forked workers inherit loaded BLAS / torch / pxr
    pools at the parent's size; the env alone reaches only libraries a worker loads later). Thread pools need none."""
    kind = _callee(call)
    if kind == "ThreadPoolExecutor":
        return True
    init = next((k.value for k in call.keywords if k.arg == "initializer"), None)
    at = 1 if kind == "Pool" else 2
    init = init if init is not None else (call.args[at] if len(call.args) > at else None)
    return init is not None and ast.unparse(init).split(".")[-1] == "worker_init"


def violations(src: str) -> list[str]:
    """Pools sized outside the owner: a pool constructor outside a function, in a function that calls neither
    capacity.sized nor (capacity.)plan, or whose size is a literal or a module constant."""
    tree = ast.parse(src)
    names = _imports(tree)
    out, inside = [], set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        nodes = list(ast.walk(fn))
        owned = any(isinstance(n, ast.Call) and _callee(n) in ("sized", "plan") for n in nodes)
        for n in nodes:
            if isinstance(n, ast.Call) and _pool_call(n, names):
                inside.add(id(n))
                size = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "max_workers"), None)
                const = isinstance(size, ast.Constant) or (isinstance(size, ast.Name) and size.id.isupper())
                if not owned or size is None or const or not _initialized(n):
                    out.append(f"{n.lineno}: {ast.unparse(n)[:80]}")
    out += [
        f"{n.lineno}: module-level {ast.unparse(n)[:80]}"
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _pool_call(n, names) and id(n) not in inside
    ]
    return sorted(set(out))


def test_every_pool_in_dexengine_is_sized_by_capacity():
    bad = {}
    for p in sorted((REPO_ROOT / "dexengine").rglob("*.py")):
        if v := violations(p.read_text(encoding="utf-8")):
            bad[str(p.relative_to(REPO_ROOT))] = v
    assert not bad, bad


@pytest.mark.parametrize(
    "src",
    [
        "from multiprocessing import Pool\ndef f(j):\n    with Pool(8) as p:\n        p.map(len, j)\n",
        "import concurrent.futures as cf\nW = 8\ndef f(j):\n    with cf.ThreadPoolExecutor(W) as ex:\n        pass\n",
        "import multiprocessing as mp\nfrom dexengine.core import capacity\n"
        "def f(j):\n    with capacity.sized('s', 2) as cap, mp.get_context('fork').Pool(4) as p:\n        pass\n",
        "from concurrent.futures import ProcessPoolExecutor\ndef f(j, n):\n    return ProcessPoolExecutor(n)\n",
        "from multiprocessing import Pool\nfrom dexengine.core import capacity\n"
        "def f(j):\n    with capacity.sized('s', len(j)) as cap, Pool(cap.workers) as p:\n        p.map(len, j)\n",
    ],
)
def test_the_structural_check_catches_planted_pools(src):
    assert violations(src)


def test_the_structural_check_passes_an_owned_pool():
    src = (
        "from multiprocessing import Pool\nfrom dexengine.core import capacity\n"
        "def f(j):\n    with capacity.sized('s', len(j)) as cap,"
        " Pool(cap.workers, capacity.worker_init, cap.initargs) as p:\n        p.map(len, j)\n"
    )
    assert not violations(src)


def _keys_in_code() -> set[str]:
    """String constants that name a NEEDS step anywhere in dexengine (literals, STEP constants, 'module:part' parts)."""
    seen = set()
    for p in (REPO_ROOT / "dexengine").rglob("*.py"):
        if p.name == "capacity.py":
            continue
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("dexengine."):
                seen.add(n.value)
            elif isinstance(n, ast.JoinedStr):
                seen.add("".join(v.value if isinstance(v, ast.Constant) else "{}" for v in n.values))
    return seen


def test_every_statement_names_a_pool_and_its_module():
    code = _keys_in_code()
    for step in C.NEEDS:
        mod, _, part = step.partition(":")
        assert (REPO_ROOT / (mod.replace(".", "/") + ".py")).is_file(), f"{step}: no such module"
        assert step in code or (part and any(k.startswith(mod) for k in code)), f"{step}: no pool names it"


def test_a_thread_pool_dies_before_its_jobs_would_import_a_native_module_first(tmp_path, monkeypatch):
    """The f0 prep pool deadlocked when two pool threads imported torch and pxr for the first time: the owner
    loads them in the main thread first and dies when one is missing. Negative: without the check the pool opens."""
    mod = tmp_path / "cap_probe_mod.py"
    mod.write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("cap_probe_mod", None)
    monkeypatch.setitem(C.NEEDS, "t.prep", C.Need(1, 1, 4, kind="thread", preload=("cap_probe_mod",)))
    monkeypatch.setattr(C, "preload", lambda mods: None)  # a preload that did not happen
    with pytest.raises(SystemExit, match="native modules"), C.sized("t.prep", 4):
        pass
    monkeypatch.setattr(C, "check_preloaded", lambda mods: None)  # the check removed: the pool opens — the test fails
    with C.sized("t.prep", 4) as cap:
        assert cap.workers >= 1
    assert "cap_probe_mod" not in sys.modules


def test_the_jobs_find_the_modules_already_loaded(tmp_path, monkeypatch):
    mod = tmp_path / "cap_probe_mod2.py"
    mod.write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("cap_probe_mod2", None)
    monkeypatch.setitem(C.NEEDS, "t.prep2", C.Need(1, 1, 4, kind="thread", preload=("cap_probe_mod2",)))
    seen = []
    with C.sized("t.prep2", 4) as cap, ThreadPoolExecutor(cap.workers) as ex:
        list(
            ex.map(lambda _i: seen.append((threading.current_thread().name, "cap_probe_mod2" in sys.modules)), range(4))
        )
    assert all(loaded for _n, loaded in seen) and len(seen) == 4


def test_the_f0_prep_pool_preloads_torch_pxr_and_the_registry_module():
    need = C.NEEDS["dexengine.retarget.f0.shadow_gpu"]
    assert need.kind == "thread"
    assert {"torch", "pxr.Usd", "registry:object_pin"} <= set(need.preload)
    from dexengine.core.locations.modules import _REGISTRY_MODULES

    assert C._module_name("registry:object_pin") == _REGISTRY_MODULES["object_pin"]


def test_the_marker_is_printed_once_per_step(monkeypatch, capsys):
    monkeypatch.setitem(C.NEEDS, "t.once", C.Need(0.1, 0.1, 2))
    monkeypatch.setattr(C, "_SAID", set())
    for _ in range(3):
        with C.sized("t.once", 2):
            pass
    err = capsys.readouterr().err
    assert err.count("[CPU] step=t.once") == 2  # start and end of the first opening
    assert "peak_parent=" in err and "peak_job=" in err


def _blas(_):
    from threadpoolctl import threadpool_info

    return [p["num_threads"] for p in threadpool_info()]


def test_worker_init_caps_the_loaded_blas_pools():
    """In a forked worker (not this process: the caps and the worker flag stay there). The caps and their reading go
    through threadpoolctl, which the retargeting venv pins (pyproject [retarget]); elsewhere it skips, naming why."""
    pytest.importorskip(
        "threadpoolctl",
        reason="threadpoolctl is not installed in this interpreter (the Isaac env) — the BLAS pool caps are measured "
        "in the retargeting venv, which pins it",
    )
    import numpy  # noqa: F401 — a BLAS pool the worker inherits loaded

    with mp.get_context("fork").Pool(1, C.worker_init, (1,)) as p:
        got = p.map(_blas, [0])[0]
    assert got and all(n == 1 for n in got)
    assert C._WORKER is None


def test_process_pool_workers_load_pxr_with_one_thread_and_the_caller_gets_its_env_back():
    """A forked worker that imports pxr reads PXR_WORK_THREAD_LIMIT (USD's TBB pool starts one thread per CPU and
    ignores OMP_NUM_THREADS): one thread there. After the pool the caller's environment is what it was, and a thread
    pool changes no thread setting (its jobs share the process's pools)."""
    import os
    import subprocess

    code = (
        "import multiprocessing as mp, os\n"
        "from dexengine.core import capacity as C\n"
        "def usd(_):\n"
        "    from pxr import Usd, Work\n"
        "    Usd.Stage.CreateInMemory()\n"
        "    return Work.GetConcurrencyLimit()\n"
        "C.NEEDS['t.usd'] = C.Need(0.1, 0.1, 2)\n"
        "C.NEEDS['t.thr'] = C.Need(0.1, 0.1, 2, kind='thread')\n"
        "with C.sized('t.usd', 2) as cap, mp.get_context('fork').Pool(cap.workers, C.worker_init, cap.initargs) as p:\n"
        "    print('workers', p.map(usd, range(2)))\n"
        "print('after', sorted(k for k in C.THREAD_ENV if k in os.environ))\n"
        "with C.sized('t.thr', 2):\n"
        "    print('thread', sorted(k for k in C.THREAD_ENV if k in os.environ))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in C.THREAD_ENV}
    env["PYTHONPATH"] = str(REPO_ROOT)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.splitlines() == ["workers [1, 1]", "after []", "thread []"]
    assert "threads=-(process-wide)" in out.stderr


def _here(_):
    return C.threads_here()


def test_threads_here_is_the_workers_threads_in_a_worker_and_the_usable_cpus_elsewhere(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # a BLAS pin (canon_env.sh) is not a limit for cKDTree queries
    monkeypatch.setattr(C, "cpu_limit", lambda dirs=None: (6, "cpu.max"))
    assert C.threads_here() == 6  # every usable CPU (not every host CPU), as workers=-1 did
    monkeypatch.setitem(C.NEEDS, "t.here", C.Need(0.1, 0.1, 2))
    with C.sized("t.here", 2) as cap, mp.get_context("fork").Pool(cap.workers, C.worker_init, cap.initargs) as p:
        assert p.map(_here, range(2)) == [1, 1]
        assert C.threads_here() == 6


def test_a_hidden_cgroup_is_not_read_as_the_host(tmp_path, monkeypatch):
    """Singularity in a Slurm job: /proc/self/cgroup names the job's cgroup, /sys/fs/cgroup does not show it."""
    proc = tmp_path / "cgroup"
    proc.write_text("0::/system.slice/job_1/step\n")
    root = tmp_path / "cg"
    root.mkdir()
    monkeypatch.setattr(M, "_host_available", lambda meminfo="": 400 * 2**30)
    kw = {"proc_cgroup": str(proc), "root": root}
    assert C.mem_limit(environ={}, **kw) == (400 * 2**30, "hidden")
    slurm = {"SLURM_JOB_ID": "7", "SLURM_MEM_PER_NODE": "20480"}
    assert C.mem_limit(environ=slurm, **kw) == (20 * 2**30, "slurm")
    per_cpu = {"SLURM_JOB_ID": "7", "SLURM_MEM_PER_CPU": "2048", "SLURM_CPUS_PER_TASK": "4"}
    assert C.mem_limit(environ=per_cpu, **kw) == (8 * 2**30, "slurm")
    assert C.mem_limit(environ={C.MEM_ENV: "6"}, **kw) == (6 * 2**30, "env")
    with pytest.raises(SystemExit, match="positive"):
        C.mem_limit(environ={C.MEM_ENV: "0"}, **kw)
    proc.write_text("4:memory:/job_1\n0::/job_1\n")  # a cgroup v1 (hybrid) host
    (root / "memory" / "job_1").mkdir(parents=True)
    (root / "memory" / "job_1" / "memory.limit_in_bytes").write_text(str(3 * 2**30))
    assert C.mem_limit(environ={}, **kw) == (3 * 2**30, "memory.v1")


def test_a_process_pool_refuses_a_hidden_limit_and_a_thread_pool_does_not(monkeypatch):
    monkeypatch.setattr(C, "cpu_limit", lambda dirs=None: (4, "affinity"))
    monkeypatch.setattr(C, "mem_limit", lambda dirs=None: (400 * 2**30, "hidden"))
    monkeypatch.setitem(C.NEEDS, "t.p", C.Need(0.5, 4.2, 8))
    monkeypatch.setitem(C.NEEDS, "t.t", C.Need(1, 1, 8, kind="thread"))
    with pytest.raises(SystemExit, match="not readable"):
        C.plan("t.p", 8)
    assert C.plan("t.t", 8).workers == 4


def test_a_slurm_job_without_a_memory_limit_in_its_cgroup_needs_the_slurm_request(tmp_path, monkeypatch):
    """A cluster job layout: its cgroup is visible (bind) and memory.max = max at every level of a --mem job."""
    rel = "system.slice/n1_slurmstepd.scope/job_1539127/step_batch/user/task_0"
    root = tmp_path / "cg"
    (root / rel).mkdir(parents=True)
    for d in [root / rel, *(root / rel).parents]:
        if d.is_relative_to(root) and d != root:
            (d / "memory.max").write_text("max\n")
    proc = tmp_path / "cgroup"
    proc.write_text(f"0::/{rel}\n")
    monkeypatch.setattr(M, "_host_available", lambda meminfo="": 400 * 2**30)
    kw = {"proc_cgroup": str(proc), "root": root}
    assert C.mem_limit(environ={}, **kw)[1] == "hidden"
    assert C.mem_limit(environ={"SLURM_JOB_ID": "1539127", "SLURM_MEM_PER_NODE": "6144"}, **kw) == (6 * 2**30, "slurm")
    (root / "system.slice/n1_slurmstepd.scope/job_1539127/memory.max").write_text(str(5 * 2**30))  # a site enforcing
    assert C.mem_limit(environ={}, **kw) == (5 * 2**30, "memory.max")


def _alloc(_):
    b = bytearray(64 * 2**20)  # a worker that really uses memory (64 MiB)
    return len(b)


def test_the_oracle_flags_a_worker_above_its_statement(monkeypatch, capsys):
    """Statement 1 MiB per job, the worker uses 64 MiB: the end of the pool prints the warning (once per step); a
    statement above the measurement prints none."""
    monkeypatch.setattr(C, "_OVER", set())
    monkeypatch.setitem(C.NEEDS, "t.small", C.Need(0.1, 0.001, 2, measured="r"))
    monkeypatch.setitem(C.NEEDS, "t.big", C.Need(0.1, 10_000.0, 2))
    for _ in range(2):
        with C.sized("t.small", 2) as cap, mp.get_context("fork").Pool(1, C.worker_init, cap.initargs) as p:
            p.map(_alloc, [0])
    with C.sized("t.big", 2) as cap, mp.get_context("fork").Pool(1, C.worker_init, cap.initargs) as p:
        p.map(_alloc, [0])
    bad = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[CPU] ✗")]
    assert len(bad) == 1 and "per_job=0.001GiB step=t.small (r)" in bad[0]


def test_a_task_empties_its_caches_and_hands_the_heap_back_whatever_it_raised(monkeypatch):
    trims = []
    monkeypatch.setattr(C.ctypes, "CDLL", lambda _: SimpleNamespace(malloc_trim=trims.append))
    shapes, kept = {"o1": 1}, {"hand": 1}
    with C.task(shapes):
        shapes["o2"] = 2
    assert shapes == {} and kept == {"hand": 1} and trims == [0]
    shapes["o3"] = 3
    with pytest.raises(RuntimeError), C.task(shapes):
        raise RuntimeError("tree failed")
    assert shapes == {} and trims == [0, 0]
