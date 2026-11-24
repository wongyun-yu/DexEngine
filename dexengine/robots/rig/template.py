"""Rig template: the MuJoCo boilerplate every hand rig shares.

`hand_rig.py` calls the element builders while it assembles a rig with ElementTree; the hand-specific parts (meshes,
bodies, palm and fingertip sites, collision geoms, finger actuators) stay in `hand_rig.py`. The committed rigs
(`assets/robots/hands/<hand>/mjcf/{right,left}.xml`) carry one marker line `<!-- clean-room U7: <key> -->` per
template line; `fill_markers` writes the template element back in place of each marker.

Every element is described once, in `_ELEMENTS` / `_DOFS` below, as (tag, ordered attributes); both the builders and
`fill_markers` read those tables, so the two users cannot drift apart.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

#: The six base DOFs of every rig root body, in contract order: (name, joint type, axis, range).
_DOFS: tuple[tuple[str, str, str, str], ...] = (
    ("pos_x", "slide", "-1 0 0", "-1.0 1.0"),
    ("pos_y", "slide", "0 0 -1", "-1.0 1.0"),
    ("pos_z", "slide", "0 -1 0", "-1.0 1.0"),
    ("rot_x", "hinge", "-1 0 0", "-6.28 6.28"),
    ("rot_y", "hinge", "0 0 -1", "-6.28 6.28"),
    ("rot_z", "hinge", "0 -1 0", "-6.28 6.28"),
)

#: Side-independent single-line elements, keyed by their marker key: (tag, ordered attributes).
_ELEMENTS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "compiler": ("compiler", (("angle", "radian"), ("meshdir", "assets"), ("autolimits", "true"))),
    "option.flag": ("flag", (("eulerdamp", "disable"),)),
    "default.geom": ("geom", (("density", "800"), ("condim", "1"), ("contype", "0"), ("conaffinity", "0"))),
    "default.position": ("position", (("kp", "300"), ("dampratio", "1.0"), ("inheritrange", "1"))),
    "default.joint": ("joint", (("damping", "0.0"), ("armature", "1.0"), ("frictionloss", "0.0"))),
    "default.site": ("site", (("size", "0.005"), ("type", "sphere"), ("rgba", "1 0 0 1"), ("group", "3"))),
    "visual.headlight": (
        "headlight",
        (("diffuse", "0.6 0.6 0.6"), ("ambient", "0.1 0.1 0.1"), ("specular", "0.9 0.9 0.9")),
    ),
    "visual.rgba": ("rgba", (("haze", "0.15 0.25 0.35 1"),)),
    "visual.global": ("global", (("azimuth", "140"), ("elevation", "-20"))),
    "worldbody.light": ("light", (("pos", "0 0 3.5"), ("dir", "0 0 -1"), ("directional", "false"))),
}

_OPTION_ATTRS = (("timestep", "0.01"), ("iterations", "10"), ("ls_iterations", "50"))
_BASE_QUAT = "0.0000000 -0.0000000 -0.7071068 0.7071068"
_BASE_ACTUATOR_KP = "1000"
_MARKER = re.compile(r"^([ \t]*)<!-- clean-room U7: ([A-Za-z0-9_.]+) -->[ \t]*$", re.MULTILINE)


def _make(tag: str, attrs: tuple[tuple[str, str], ...]) -> ET.Element:
    """An unattached element whose attributes keep the given order."""
    el = ET.Element(tag)
    for k, v in attrs:
        el.set(k, v)
    return el


def _sub(parent: ET.Element, tag: str, attrs: tuple[tuple[str, str], ...]) -> ET.Element:
    el = _make(tag, attrs)
    parent.append(el)
    return el


def _keyed(key: str) -> ET.Element:
    tag, attrs = _ELEMENTS[key]
    return _make(tag, attrs)


def _base_actuator(side: str, dof: str) -> ET.Element:
    return _make(
        "position", (("name", f"{side}_{dof}_position"), ("joint", f"{side}_{dof}"), ("kp", _BASE_ACTUATOR_KP))
    )


def add_skeleton(root: ET.Element) -> None:
    """Append the compiler, option (with its flag), default and visual blocks under the `<mujoco>` root."""
    root.append(_keyed("compiler"))
    option = _sub(root, "option", _OPTION_ATTRS)
    option.append(_keyed("option.flag"))
    default = _sub(root, "default", ())
    for key in ("default.geom", "default.position", "default.joint", "default.site"):
        default.append(_keyed(key))
    visual = _sub(root, "visual", ())
    for key in ("visual.headlight", "visual.rgba", "visual.global"):
        visual.append(_keyed(key))


def add_world_assets(asset: ET.Element, side: str) -> None:
    """Append the skybox texture and the side's ground-plane texture and material under `<asset>`."""
    _sub(
        asset,
        "texture",
        (
            ("type", "skybox"),
            ("builtin", "gradient"),
            ("rgb1", "1 1 1"),
            ("rgb2", "1 1 1"),
            ("width", "512"),
            ("height", "3072"),
        ),
    )
    _sub(
        asset,
        "texture",
        (
            ("type", "2d"),
            ("name", f"{side}_groundplane"),
            ("builtin", "checker"),
            ("mark", "edge"),
            ("rgb1", "0.2 0.3 0.4"),
            ("rgb2", "0.1 0.2 0.3"),
            ("markrgb", "0.8 0.8 0.8"),
            ("width", "300"),
            ("height", "300"),
        ),
    )
    _sub(
        asset,
        "material",
        (
            ("name", f"{side}_groundplane"),
            ("texture", f"{side}_groundplane"),
            ("texuniform", "true"),
            ("texrepeat", "5 5"),
            ("reflectance", "0.2"),
        ),
    )


def add_light(worldbody: ET.Element) -> None:
    """Append the overhead light under `<worldbody>`."""
    worldbody.append(_keyed("worldbody.light"))


def base_body_quat() -> str:
    """`quat` attribute of the synthesised hand root body (makes the base DOFs move along world x, y, z)."""
    return _BASE_QUAT


def base_head(side: str) -> list[ET.Element]:
    """Elements to insert first into the hand root body: the tracking camera, then the six base joints."""
    head = [_make("camera", (("name", f"{side}_track"), ("pos", "0 0.4 -0.2"), ("xyaxes", "1 0 0 0 -0.447 0.894")))]
    for dof, jtype, axis, rng in _DOFS:
        head.append(
            _make(
                "joint", (("name", f"{side}_{dof}"), ("type", jtype), ("pos", "0 0 0"), ("axis", axis), ("range", rng))
            )
        )
    return head


def add_base_actuators(actuator: ET.Element, side: str) -> None:
    """Append the six base position actuators (they come first in `<actuator>`)."""
    for dof, *_ in _DOFS:
        actuator.append(_base_actuator(side, dof))


def _marker_element(key: str, side: str) -> ET.Element:
    if key in _ELEMENTS:
        return _keyed(key)
    prefix = "actuator.base."
    if key.startswith(prefix) and key[len(prefix) :] in {d[0] for d in _DOFS}:
        return _base_actuator(side, key[len(prefix) :])
    raise KeyError(f"unknown rig template marker key {key!r}")


def fill_markers(xml_text: str, side: str) -> str:
    """Replace every `<!-- clean-room U7: <key> -->` marker line of a committed rig file by its template element.

    The marker's indentation and line ending are kept; nothing else in the text changes. An unknown key raises.
    """

    def _sub_marker(m: re.Match) -> str:
        return m.group(1) + ET.tostring(_marker_element(m.group(2), side), encoding="unicode")

    return _MARKER.sub(_sub_marker, xml_text)
