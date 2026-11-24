"""Build side of rule S-carried for the extra slots: the carry statement.

Only rows of kind carried_attached (written by `dexengine/data/objects/extra_slots.py`) change anything here; a run
without one writes no statement (training and judge cfgs byte-identical). A carried slot is the kinematic asset of
every slot; its pose follows the simulated carrier for the whole clip (`scene.extra_carry`).
Statement `cfg.extra_slot_carry` (clip × slot, None for other cells) = {carrier_role, carrier_side, carrier_slot (the
carrier's slot index in the same clip's cells), rel}.
"""

from __future__ import annotations


def carry_statement(cfg, cells: list, K: int) -> None:
    """Write `cfg.extra_slot_carry` when a cell has a carry row (else nothing — the cfg stays as it was)."""
    if not any(x is not None and x.get("carry") for r in cells for x in r):
        return
    st = []
    for c, r in enumerate(cells):
        row = [None] * K
        names = {x["name"]: k for k, x in enumerate(r) if x is not None}
        for k, x in enumerate(r):
            if x is None or not x.get("carry"):
                continue
            cy = x["carry"]
            kc = None
            if cy["carrier_role"] == "slot":
                if cy["carrier"] not in names:
                    raise RuntimeError(
                        f"[EXTRA_CARRY] clip cell {c} slot {k} ({x['name']}): carrier slot "
                        f"{cy['carrier']} does not stand in this clip"
                    )
                kc = names[cy["carrier"]]
            row[k] = {
                "carrier_role": cy["carrier_role"],
                "carrier_side": cy["carrier_side"],
                "carrier_slot": kc,
                "rel": list(cy["rel"]),
            }
        st.append(row)
    cfg.extra_slot_carry = st
