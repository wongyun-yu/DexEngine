"""Single window onto PhysX CPU properties: the object material and mass caches and the articulation COM cache.

``get_material_properties()``, ``get_masses()`` and ``get_coms()`` of the PhysX tensor API are slow calls that gather
every env's values to the CPU. The privileged observations read them every step, the reset randomisation at every
reset (some envs reset every step), and Isaac Lab's ``ArticulationData.body_com_pose_b`` repeated the call every new
physics step: about 20% of the training loop's wall time.

Values change only through this module's :func:`write` (object material and mass). Nothing writes the robot
articulation's COMs (COM randomisation touches objects only, so the COM cache goes on the robot articulations only).
A cache fills once from PhysX on its first read, and a write updates the same rows of PhysX and the cache with the same
values, so the cache is bit-identical with PhysX, with **one exception**: the PhysX tensor API shares materials by a
6-decimal key (:func:`material_key`), so when a material with the same key is alive the earlier float is attached
instead of the written one (difference < 1e-6). That cell holds the written (intended) value until the next oracle
comparison, which adopts the PhysX value (marker :data:`MARK_ALIAS`).

Oracle: every :data:`ORACLE_EVERY`-th read of a cache re-reads PhysX and **compares bit for bit**, failing at once
on a difference (it means something wrote outside this window). The only accepted difference is what the key rule
above can make: when **every** mismatching material cell has the same key in PhysX and the cache, it adopts them and
goes on (:func:`_adopt_key_aliases`); a single cell with a different key fails as before (decided by key, not by size:
a 1-ulp difference across a key boundary fails too). Passing leaves a ``[PHYSX_CACHE]`` marker: bit match =
:data:`MARK_MATCH`, alias adopted = :data:`MARK_ALIAS` (counted separately, so the match count never includes
aliases), failure = :data:`MARK_DIE`. A log scanner reads these constants by AST.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

#: Oracle period (reads). Each cache re-reads PhysX about once every 250 steps; the cost is negligible.
ORACLE_EVERY = 500

#: Log markers: these constants own the wording. A reader runs without Isaac and does not import this module; it reads
#: these assignments by AST (no second copy of the wording).
MARK = "[PHYSX_CACHE]"
MARK_MATCH = "oracle match"
MARK_ALIAS = "material key alias"
MARK_DIE = "cache ≠ PhysX"

_GET = {"material": "get_material_properties", "mass": "get_masses"}
_SET = {"material": "set_material_properties", "mass": "set_masses"}
#: Does PhysX return the written value as is? Not for mass: PhysX stores its inverse and is 1 ulp off (oracle on
#: a run: max difference 1.5e-8). Such kinds re-read PhysX after a write to fill the cache (the mass table is one
#: cell per env, so it is cheap). Materials are as written with one exception: the PhysX material key alias
#: (:func:`material_key` · :func:`_adopt_key_aliases`).
_ROUNDTRIP_EXACT = {"material": True, "mass": False}


def material_key(t) -> str:
    """Material dictionary key of the PhysX tensor API (omni.physx.tensors 107.3); this function is the rule's one home.

    ``set_material_properties`` promotes each shape's three float32 (static, dynamic, restitution) to double, builds the
    key with C ``"%.6f_%.6f_%.6f"`` and, when a material with that key is alive anywhere in the simulation, **attaches
    that material without changing its values** (it lives by reference count and leaves the dictionary at 0). So
    writing another float equal to 6 decimals gets the earlier float back (difference < 1e-6). Python's ``.6f`` rounds
    exactly like C printf ``%.6f`` (exact decimal expansion of the binary value, ties to even); a self-test compares it
    with libc ``snprintf`` on boundary values. ``t`` = the three components (a tensor row or a sequence).
    Basis: reading the binary (``.so`` strings ``%.6f`` and ``setMaterialProperties``) plus an independent
    reproduction (a simulator bit-matching 122,400 writes)."""
    return "_".join(f"{float(x):.6f}" for x in t)


def _say_oracle(name: str, n_ok: int, reads: int) -> None:
    if n_ok in (1, 10) or n_ok % 100 == 0:
        print(f"{MARK} {name} {MARK_MATCH} {n_ok} times (reads {reads})", flush=True)


def _slot(env, asset_name: str, kind: str) -> dict:
    store = env.__dict__.setdefault("_physx_prop_cache", {})
    key = (asset_name, kind)
    slot = store.get(key)
    if slot is None:
        view = env.scene[asset_name].root_physx_view
        cpu = getattr(view, _GET[kind])().clone()
        # The material table is padded to max_shapes cells because envs have different shape counts. PhysX drops what is
        # written to a cell without a shape and returns 0, so the first read records every all-zero shape cell as
        # padding
        # and the cache keeps it at 0. (A real shape starting with a zero material would break this assumption; the
        # oracle then fails, so it is never silently wrong.)
        pad = (cpu == 0).all(dim=-1, keepdim=True) if kind == "material" else None
        slot = {"cpu": cpu, "dev": None, "reads": 0, "ok": 0, "pad": pad}
        store[key] = slot
        npad = int(pad.sum()) if pad is not None else 0
        print(
            f"{MARK} {asset_name}.{kind} cache start shape={tuple(cpu.shape)} padding cells {npad} · "
            f"bit comparison every {ORACLE_EVERY} reads",
            flush=True,
        )
    return slot


def _count_read(env, asset_name: str, kind: str, slot: dict) -> None:
    slot["reads"] += 1
    if slot["reads"] % ORACLE_EVERY:
        return
    fresh = getattr(env.scene[asset_name].root_physx_view, _GET[kind])()
    if not torch.equal(fresh, slot["cpu"]):
        if kind == "material" and _adopt_key_aliases(asset_name, slot, fresh):
            return  # alias adopted: not counted as a match (slot["ok"]); MARK_ALIAS is logged separately
        bad = fresh != slot["cpu"]
        idx = bad.nonzero()[:4].tolist()
        diff = float((fresh.double() - slot["cpu"].double()).abs().max())
        raise RuntimeError(
            f"{MARK} {asset_name}.{kind} {MARK_DIE} (max difference {diff:.3e}, reads {slot['reads']}, "
            f"mismatching elements {int(bad.sum())} · of which PhysX value 0 {int((bad & (fresh == 0)).sum())} · "
            f"e.g. {[(i, float(fresh[tuple(i)]), float(slot['cpu'][tuple(i)])) for i in idx]}) — "
            "something wrote to PhysX outside this window (physx_props.write), or the cache does not know a PhysX "
            "storage rule"
        )
    slot["ok"] += 1
    _say_oracle(f"{asset_name}.{kind}", slot["ok"], slot["reads"])


def _adopt_key_aliases(asset_name: str, slot: dict, fresh: torch.Tensor) -> bool:
    """Is the oracle mismatch only the PhysX material key alias? When **every** mismatching shape cell has the same
    :func:`material_key` in PhysX and the cache, those cells of the cache (CPU and device mirror) adopt the PhysX value,
    the :data:`MARK_ALIAS` marker prints, and it returns True. With any cell of a different key nothing changes and it
    returns False (the caller fails; no partial fix). A cell with a non-finite value (NaN, inf) is no alias, even with
    the same key text (NaN differs from itself and would trip every comparison; it is a defect)."""
    cells = (fresh != slot["cpu"]).any(dim=-1).nonzero()
    e, s = cells[:, 0], cells[:, 1]
    ph, ca = fresh[e, s], slot["cpu"][e, s]
    if not (bool(torch.isfinite(ph).all()) and bool(torch.isfinite(ca).all())):
        return False
    if not all(material_key(a) == material_key(b) for a, b in zip(ph.tolist(), ca.tolist(), strict=True)):
        return False
    slot["cpu"][e, s] = ph
    if slot["dev"] is not None:
        dev = slot["dev"]
        dev[e.to(dev.device), s.to(dev.device)] = ph.to(dev.device)
    slot["alias_reads"] = slot.get("alias_reads", 0) + 1
    slot["alias_cells"] = slot.get("alias_cells", 0) + len(cells)
    print(
        f"{MARK} {asset_name}.material {MARK_ALIAS} {len(cells)} cells (PhysX dictionary %.6f key) — cache takes the "
        f"PhysX values (reads {slot['reads']} · alias comparisons so far {slot['alias_reads']} · cells so far "
        f"{slot['alias_cells']})",
        flush=True,
    )
    return True


def read(env, asset_name: str, kind: str) -> torch.Tensor:
    """CPU **copy** of every env's values: the caller may modify it, the cache stays. Bit-identical with PhysX, except
    material key alias cells (difference < 1e-6, module header), which hold the written (intended) value until the next
    oracle comparison adopts the PhysX value."""
    slot = _slot(env, asset_name, kind)
    _count_read(env, asset_name, kind, slot)
    return slot["cpu"].clone()


def read_device(env, asset_name: str, kind: str, device) -> torch.Tensor:
    """Device mirror of every env's values: **read only** (no in-place edits). Writes refresh the changed rows only."""
    slot = _slot(env, asset_name, kind)
    _count_read(env, asset_name, kind, slot)
    dev = slot["dev"]
    if dev is None or dev.device != torch.device(device):
        dev = slot["dev"] = slot["cpu"].to(device)
    return dev


def write(env, asset_name: str, kind: str, values: torch.Tensor, env_ids_cpu: torch.Tensor) -> None:
    """Write the ``env_ids_cpu`` rows of ``values`` to PhysX and update the same rows of the cache and device mirror."""
    slot = _slot(env, asset_name, kind)
    view = env.scene[asset_name].root_physx_view
    getattr(view, _SET[kind])(values, env_ids_cpu)
    if not _ROUNDTRIP_EXACT[kind]:
        slot["cpu"] = getattr(view, _GET[kind])().clone()
        if slot["dev"] is not None:
            slot["dev"] = slot["cpu"].to(slot["dev"].device)
        return
    rows = values[env_ids_cpu]
    if slot["pad"] is not None:  # PhysX drops padding cells: 0 in the cache too
        rows = rows.masked_fill(slot["pad"][env_ids_cpu], 0)
    slot["cpu"][env_ids_cpu] = rows
    if slot["dev"] is not None:
        dev = slot["dev"]
        dev[env_ids_cpu.to(dev.device)] = rows.to(dev.device)


def install_articulation_com_cache(env, robots) -> None:
    """Install the COM cache on the **robot** articulations (arm and hand) only: ``data.body_com_pose_b`` is read once
    and kept.

    Isaac Lab re-reads this property with ``get_coms()`` (CPU) every new physics step. Nothing writes the robot COMs,
    so the value never changes; an oracle with the same period compares it bit for bit with PhysX. Objects
    (articulated ones included) are left out: COM randomisation changes the object COMs at every reset. ``robots`` =
    the robot articulation objects (each once).
    """
    seen = set()
    for art in robots:
        data = art.data
        if id(data) in seen or getattr(type(data), "_dexengine_com_cached", False):
            continue
        seen.add(id(data))
        data.__class__ = _com_cached_class(type(data))
        print(
            f"{MARK} robot articulation {art.cfg.prim_path}.body_com_pose_b cache installed · "
            f"bit comparison every {ORACLE_EVERY} reads",
            flush=True,
        )


_COM_CLASSES: dict = {}


def _com_cached_class(base: type) -> type:
    cls = _COM_CLASSES.get(base)
    if cls is not None:
        return cls

    class _ComCached(base):
        _dexengine_com_cached = True

        @property
        def body_com_pose_b(self) -> torch.Tensor:
            buf = self._body_com_pose_b
            if buf.timestamp < self._sim_timestamp:
                st = self.__dict__
                st["_dexengine_com_reads"] = n = st.get("_dexengine_com_reads", 0) + 1
                cache = st.get("_dexengine_com_cache")
                if cache is None or n % ORACLE_EVERY == 0:
                    # the same formula as Isaac Lab (articulation_data.body_com_pose_b)
                    pose = self._root_physx_view.get_coms().to(self.device)
                    pose[..., 3:7] = math_utils.convert_quat(pose[..., 3:7], to="wxyz")
                    if cache is not None:
                        if not torch.equal(pose, cache):
                            raise RuntimeError(
                                f"{MARK} articulation body_com_pose_b {MARK_DIE} — "
                                "something writes an articulation COM now. "
                                "Add a window that invalidates the cache first"
                            )
                        st["_dexengine_com_ok"] = ok = st.get("_dexengine_com_ok", 0) + 1
                        _say_oracle("articulation.body_com_pose_b", ok, n)
                    st["_dexengine_com_cache"] = cache = pose
                buf.data = cache
                buf.timestamp = self._sim_timestamp
            return buf.data

    _ComCached.__name__ = f"ComCached{base.__name__}"
    _COM_CLASSES[base] = _ComCached
    return _ComCached
