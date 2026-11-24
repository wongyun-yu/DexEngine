"""Articulated assets: the asset id of a paired articulation (object_asset_id and its inverse), the self-collision
statement per joint kind and the meshes of a link — re-exported by core.artifact_paths (their import path)."""

from __future__ import annotations

#: Separator of paired-articulation asset ids — an object id containing it would read as two identities, so it dies.
ASSET_SEP = "__"
_ARTIC_KINDS = ("hinge", "slide")


def object_asset_id(obj_id, articulation: dict | None = None) -> str:
    """Object asset id — the key of the `usd_file` · `mesh_dir` (surface points · keypoints) paths. Rule (not a name
    table):
      · used as a rigid body (no articulation), or articulated with the moving part inside that object's folder (no
        `parts.moving_id` — arctic `top.obj`) → `obj_id`.
      · the moving part is **another object** (`parts.moving_id` — D1-g, two OakInk2 rigid bodies joined)
        → `<obj_id>__<moving_id>__<kind>` · parts fixed to a link (`parts.fixed`) append their ids (sorted)
        `<obj_id>__<moving_id>__<kind>__<fixed id>…` (body+lever ≠ body alone — a different mesh).
        The same base id is used as a rigid body and paired with other parts or kinds — sharing one file would load
        the wrong part and the wrong point cloud.
    An unknown joint kind · an empty moving_id · an id containing the separator · a paired id starting or ending with
    '_' (`a_`+`b` and `a`+`_b` would join the same — not injective) die."""
    oid = str(obj_id)
    if not oid or ASSET_SEP in oid:
        raise ValueError(f"object id {oid!r} — empty or contains the asset separator {ASSET_SEP!r}")
    if not articulation:
        return oid
    kind = articulation.get("type")
    if kind not in _ARTIC_KINDS:
        raise ValueError(f"{oid}: joint kind {kind!r} — only {_ARTIC_KINDS} are known")
    parts = articulation.get("parts") or {}
    if "moving_id" not in parts:
        return oid
    mid = str(parts["moving_id"] or "")
    if not mid or ASSET_SEP in mid:
        raise ValueError(f"{oid}: moving part id {mid!r} — empty or contains the asset separator")
    if "_" in (oid[0], oid[-1], mid[0], mid[-1]):
        raise ValueError(
            f"paired ids {oid!r}·{mid!r} — starting or ending with '_' makes the join {ASSET_SEP!r} non-injective"
        )
    fids = sorted(str(f.get("id") or "") for f in parts.get("fixed") or [])
    for x in fids:  # fixed-part ids follow the same join rule
        if not x or ASSET_SEP in x or "_" in (x[0], x[-1]):
            raise ValueError(f"{oid}: fixed part id {x!r} — empty, contains the separator, or starts/ends with '_'")
    if len(set(fids)) != len(fids) or {oid, mid} & set(fids):
        raise ValueError(f"{oid}: fixed part ids {fids} overlap (the same part twice · equal to a link part)")
    return ASSET_SEP.join((oid, mid, kind, *fids))


#: Articulation self-collision — do **parts (links) of the same articulation** collide with each other. The joint kind
#: decides (not an object name table). Collisions with other objects, hands and the desk are outside this statement
#: (always on). Readers: the bake (USD articulation root attribute) · env spawn (articulation_props).
#:   slide  off — inner drawer ↔ outer body on rails. The convex-decomposed body fills the drawer's space, so with it
#:             on they push each other apart. The drawer's travel is limited by the joint limits (spec lo · hi), not
#:             by collision.
#:   hinge  off — the value env spawn has always used for every kind.
ARTIC_SELF_COLLISIONS = {"hinge": False, "slide": False}
#: Kinds whose USD authors the value itself (opening the asset alone gives the same answer). USDs of other kinds do
#: not author the attribute (bake output as is) and only env spawn applies the value. Expiry: when the hinge assets
#: are rebaked, widen this to all kinds and delete this line.
ARTIC_SELF_COLLISIONS_BAKED = ("slide",)
#: The PhysX articulation root's self-collision attribute · the API schema that makes PhysX read it
ARTIC_SELF_COLLISIONS_ATTR = "physxArticulation:enabledSelfCollisions"
ARTIC_SELF_COLLISIONS_API = "PhysxArticulationAPI"


def artic_self_collisions(kind: str) -> bool:
    """Joint kind → are collisions between parts of the same articulation on. An unknown kind dies."""
    if kind not in _ARTIC_KINDS or kind not in ARTIC_SELF_COLLISIONS:
        raise ValueError(
            f"joint kind {kind!r} — no self-collision statement (only {sorted(ARTIC_SELF_COLLISIONS)} are known)"
        )
    return bool(ARTIC_SELF_COLLISIONS[kind])


#: Articulation links — per link one main part (parts[link]) + zero or more fixed parts (parts.fixed[].link == link)
ARTIC_LINKS = ("base", "moving")


def artic_link_files(parts: dict, link: str) -> list[tuple[str, dict | None]]:
    """The meshes making up articulation link `link` — [(path relative to mesh_dir, link-frame pose {pos, quat_wxyz}
    | None)]. First = the link's main part parts[link] (no pose = the link frame itself) · the rest = parts fixed to
    that link (parts.fixed — one-body parts the D1-g pairing rule found in the data, e.g. a lever on a microwave body)
    — the owner of how to read them. The bake, surface points, first-frame geometry and placement vertices all read
    this list (reading only the link part would silently drop the fixed parts).
    An unknown link or a fixed part without a pose dies."""
    if link not in ARTIC_LINKS:
        raise ValueError(f"articulation link {link!r} — only {ARTIC_LINKS} are known")
    out: list[tuple[str, dict | None]] = [(parts[link], None)]
    for f in parts.get("fixed") or []:
        if f.get("link") not in ARTIC_LINKS or not f.get("mesh") or not (f.get("pose") or {}).get("quat_wxyz"):
            raise ValueError(f"fixed part entry {f!r} — needs link ∈ {ARTIC_LINKS} · mesh · pose(pos, quat_wxyz)")
        if f["link"] == link:
            out.append((f["mesh"], f["pose"]))
    return out
