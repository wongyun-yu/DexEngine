"""Proves the MPS contract of a judge lane **inside the judge process**.

Why here. Even when the lane launcher sets the pipe directory, without a daemon CUDA
**silently runs non-MPS** (measured on a 48 GB judge server) — the lane finishes fine and only the throughput drops to
time-slicing level. That fact is confirmed only by this process being in the client list of an MPS server.

Contract (environment — the value owner is the lane launcher):
  DEXENGINE_JUDGE_MPS=on   after creating the CUDA context ① this device's UUID == DEXENGINE_JUDGE_MPS_UUID (device
                         restriction) ② this pid is in the client list of one of the servers of the control daemon
                         (CUDA_MPS_PIPE_DIRECTORY). If either fails, SystemExit — that lane is invalid.
  DEXENGINE_JUDGE_MPS=off or unset  if CUDA_MPS_PIPE_DIRECTORY is set, it is MPS outside the contract and dies.
  Otherwise marker only.
Markers: `[JUDGE_MPS] gpu=<N> uuid=<…> pipe=<…> lanes=<K> server=<pid> client=<pid>` · `[JUDGE_MPS] off`.
No effect on the judge output (one 1-element tensor to confirm the context — does not touch the RNG state).

Lane VRAM budget (lane_cap — assert_lane calls it at the end of both branches):
  DEXENGINE_JUDGE_LANE_MIB=<MiB>  the share this lane uses on the GPU.
      Value owner = the lane launcher (GPU total ÷ lane count).
      The torch cap = share − NONTORCH_MIB, set with set_per_process_memory_fraction.
      Above it the allocator **first returns its own cache** and retries
      (kernels·batch sizes unchanged → judge output unchanged · 20-clip A/B byte-equal).
      Required with MPS on (dies without it), and the MPS wall
      CUDA_MPS_PINNED_DEVICE_MEM_LIMIT must be `<UUID>=<share>M`.
      Given outside MPS (an A/B run) it sets only the torch cap (a wall there is outside the contract and dies).
  Markers: `[VRAM_CAP] lane on lane_mib=<share> …` · `[VRAM_CAP] lane off …`.
  stage2_ground.sh looks for this line with the same value in the log of a judging it gave a share to
  (missing → that judging is invalid — an input with no consumer).
  The proof that the MPS wall is set is not re-reading env but **a value this process measures**. The MPS wall
  does not change the cudaMemGetInfo total (an RTX 5090 measured total 32106 unchanged) but trims **only the free
  amount** to wall − own use (wall 15998M → free 15416 · 2000M → 1503). So:
    ① free (process view) > share means the wall is not set → dies.
    ② if the device view (nvidia-smi --id=<UUID> memory.free · outside MPS) exceeds the process view by WALL_GAP_MIB or
       more, only the wall trimmed the free amount → `wall_proof=measured`. If a neighbour fills the GPU so the device
       view is small too, wall and physical limit cannot be told apart →
       `wall_proof=undecided(…)` (does not die — ① passed). The first wave (lanes started on an empty GPU) must be
       measured.
"""

from __future__ import annotations

import os
import subprocess

# VRAM outside torch in a judge process (CUDA·MPS client context · cuBLAS/cuSOLVER workspaces · graph executables).
# The lane share minus this is the torch cap. Basis: of 7 MPS judge OOMs on a 48 GB judge server,
# 'process use − allocated − reserved·unallocated' = 2.8–4.1 GiB (max 4188 MiB, hot3d_00) · an RTX 5090 without MPS
# 1.2–2.2 GB · MPS client context ≈ 500 MiB inside the wall (RTX 5090 measured). Max + margin.
NONTORCH_MIB = 4608
# Floor of the torch cap — live tensors of a heavy clip + one request must fit. Basis: of 7 MPS judge OOMs on a 48 GB
# judge server, max = taco_06 allocated 7.13 GiB + request 1.01 GiB = 8335 MiB. Below the floor that clip OOMs itself
# every time and the chunk ends exhausted — 4 lanes on a 48 GB GPU (cap ≈ 7390) do that. 3 lanes (≈ 11390) are above.
TORCH_FLOOR_MIB = 8448
# Threshold of wall proof ② — the device-view minus process-view gap must be at least this to read 'the wall trimmed
# it'. Larger than the gap from a neighbour releasing a transient batch between the two readings (≈ 0.1 s):
# a value above the high-water mark of one judge transient batch (post-check 200000 poses) ≈ 4 GiB.
WALL_GAP_MIB = 4608


def _ctl(pipe: str, cmd: str) -> list[str]:
    r = subprocess.run(
        ["nvidia-cuda-mps-control"],
        input=cmd + "\n",
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe},
    )
    if r.returncode != 0:
        raise SystemExit(
            f"[JUDGE_MPS] ✗ control daemon does not respond (pipe={pipe} rc={r.returncode}: "
            f"{(r.stdout or r.stderr).strip()[:120]}) — only the pipe was set, no daemon"
        )
    return r.stdout.split()


def _dev_free_mib(uuid: str) -> int | None:
    """Free VRAM (MiB) in the device view — nvidia-smi is not an MPS client and sees the GPU outside the wall. Read
    failure = None."""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--id={uuid}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    s = r.stdout.strip()
    return int(s) if r.returncode == 0 and s.isdigit() else None


def assert_lane(torch, dev: str) -> dict | None:
    """Called right after the judge main picks the CUDA device. SystemExit on contract violation. Returns the lane_cap
    result."""
    mode = os.environ.get("DEXENGINE_JUDGE_MPS", "")
    pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "")
    if mode != "on":
        if mode not in ("", "off"):
            raise SystemExit(f"[JUDGE_MPS] ✗ DEXENGINE_JUDGE_MPS={mode!r} — on|off only")
        if pipe:
            raise SystemExit(
                f"[JUDGE_MPS] ✗ CUDA_MPS_PIPE_DIRECTORY={pipe} is set but DEXENGINE_JUDGE_MPS is "
                "not on — MPS outside the contract (did not go through the lane launcher)"
            )
        print(f"[JUDGE_MPS] off{'' if mode else '(unset)'}", flush=True)
        return lane_cap(torch, dev, mps_uuid=None)
    if not pipe or dev != "cuda":
        raise SystemExit(f"[JUDGE_MPS] ✗ on but pipe={pipe or 'none'} dev={dev} — cannot be an MPS lane")
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    want = os.environ.get("DEXENGINE_JUDGE_MPS_UUID", "")
    if uuid != want:
        raise SystemExit(
            f"[JUDGE_MPS] ✗ device UUID {uuid} ≠ the launcher's {want or 'none'} (device restriction violated)"
        )
    me = str(os.getpid())
    servers = _ctl(pipe, "get_server_list")
    for s in servers:
        if me in _ctl(pipe, f"get_client_list {s}"):
            print(
                f"[JUDGE_MPS] gpu={os.environ.get('DEXENGINE_JUDGE_MPS_GPU', '?')} uuid={uuid} pipe={pipe} "
                f"lanes={os.environ.get('DEXENGINE_JUDGE_MPS_LANES', '?')} server={s} client={me}",
                flush=True,
            )
            return lane_cap(torch, dev, mps_uuid=uuid)
    raise SystemExit(
        f"[JUDGE_MPS] ✗ pid {me} is not in the MPS client list (servers {servers or 'none'}) — "
        "CUDA fell back to non-MPS. This lane is invalid"
    )


def lane_cap(torch, dev: str, mps_uuid: str | None) -> dict | None:
    """Applies the lane VRAM budget to this process (module header). mps_uuid = that device UUID for an MPS lane, else
    None. Returns {lane_mib, torch_mib, frac} or None (no budget). SystemExit on contract violation."""
    env = os.environ.get("DEXENGINE_JUDGE_LANE_MIB", "")
    wall = os.environ.get("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", "")
    if not env:
        if mps_uuid is not None:
            raise SystemExit(
                "[VRAM_CAP] ✗ MPS lane but DEXENGINE_JUDGE_LANE_MIB is not set — lanes on the same GPU "
                "fight for VRAM without shares and OOM-kill their neighbours. "
                "start the lane through the lane launcher"
            )
        if wall:
            raise SystemExit(
                f"[VRAM_CAP] ✗ CUDA_MPS_PINNED_DEVICE_MEM_LIMIT={wall} is set but there is no lane budget — "
                "a wall outside the contract"
            )
        print("[VRAM_CAP] lane off (DEXENGINE_JUDGE_LANE_MIB unset — no per-process VRAM cap)", flush=True)
        return None
    if not env.isdigit() or int(env) <= 0:
        raise SystemExit(f"[VRAM_CAP] ✗ DEXENGINE_JUDGE_LANE_MIB={env!r} — must be a positive MiB integer")
    if dev != "cuda":
        raise SystemExit(f"[VRAM_CAP] ✗ lane budget {env} MiB given but dev={dev} — not a CUDA judging")
    lane = int(env)
    torch_mib = lane - NONTORCH_MIB
    if torch_mib < TORCH_FLOOR_MIB:
        raise SystemExit(
            f"[VRAM_CAP] ✗ torch cap {torch_mib} MiB (lane budget {lane} − outside torch {NONTORCH_MIB}) < floor "
            f"{TORCH_FLOOR_MIB} MiB — too many lanes (heavy clips OOM every time · a 48 GB GPU takes 3 lanes)"
        )
    want = f"{mps_uuid}={lane}M" if mps_uuid is not None else ""
    if wall != want:
        raise SystemExit(
            f"[VRAM_CAP] ✗ MPS wall CUDA_MPS_PINNED_DEVICE_MEM_LIMIT={wall or 'none'} ≠ "
            f"{want or 'none (outside MPS)'} — the lane budget and the wall come from one value (the lane launcher)"
        )
    free_b, total_b = torch.cuda.mem_get_info()
    frac = torch_mib * 2**20 / total_b  # denominator = the cudaMemGetInfo total the allocator uses for the fraction
    if not 0.0 < frac <= 1.0:
        raise SystemExit(
            f"[VRAM_CAP] ✗ torch cap {torch_mib} MiB / device total {total_b >> 20} MiB = {frac:.4f} — outside (0, 1]"
        )
    torch.cuda.set_per_process_memory_fraction(frac)
    rb = torch.cuda.get_per_process_memory_fraction()
    if abs(rb - frac) > 1e-9:  # the proof it is set is the value read back from the allocator
        raise SystemExit(
            f"[VRAM_CAP] ✗ value read back after set_per_process_memory_fraction({frac:.6f}) is {rb:.6f} "
            "— the cap was not set"
        )
    proof = wall_proof(lane, free_b >> 20, mps_uuid)
    print(
        f"[VRAM_CAP] lane on lane_mib={lane} nontorch_mib={NONTORCH_MIB} torch_mib={torch_mib} "
        f"total_mib={total_b >> 20} frac={frac:.6f} readback={rb:.6f} free_mib={free_b >> 20} "
        f"wall={wall or '-'} wall_proof={proof}",
        flush=True,
    )
    return {"lane_mib": lane, "torch_mib": torch_mib, "frac": frac, "wall_proof": proof}


def wall_proof(lane: int, proc_free: int, mps_uuid: str | None) -> str:
    """Measurement that the MPS wall is set on this process (module header ①·②). Returns the marker value. SystemExit
    if the wall is not set."""
    if mps_uuid is None:
        return "-"
    if proc_free > lane:
        raise SystemExit(
            f"[VRAM_CAP] ✗ MPS wall not set — free VRAM seen by this process {proc_free} MiB > "
            f"lane share {lane} MiB (with the wall set, free ≤ wall − own use). "
            "the wall in env must exist before the CUDA context is created"
        )
    dev_free = _dev_free_mib(mps_uuid)
    if dev_free is None:
        return f"undecided(smi_fail proc_free={proc_free})"
    if dev_free - proc_free >= WALL_GAP_MIB:
        return f"measured(dev_free={dev_free}>proc_free={proc_free})"
    return f"undecided(dev_free={dev_free} proc_free={proc_free} gap<{WALL_GAP_MIB})"
