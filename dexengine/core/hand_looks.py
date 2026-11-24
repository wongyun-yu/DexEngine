"""Render appearance of each hand (the manufacturer's official look) — the single owner of the values.

The render-body generator (dexengine/robots/usd/render_body.py) paints the hand visual meshes by this declaration
(dexengine/robots/usd/hand_look.py apply), and the oracle in the same module (problems) measures the output USD
against it. The training plate (collision, joints, mass) is untouched — only visual materials of the render plate
change (cross-check: dexengine/robots/usd/render_plate.py parity).

Rules (per hand, in order, first match wins): (link regex, material regex, paint). Link names are matched with the
side removed (the word `right`/`left` → `{side}`); the material name is the prim name of the material bound to the
mesh ("" = none bound). Paint None keeps the original colour (a part that already matches the official look). A
hand-link (`{side}_…`) mesh that matches no rule makes the generator and the oracle raise (an unseen part is never
passed silently). Hand parts merged into the wrist (EE) link are painted only when a rule matches (the rest is arm);
a rule names that link by the token EE_TOKEN (``\\{ee\\}`` in the regex), which the painter replaces with the arm's EE
link (ArmSpec.ee_body_name, side removed) — the look stays a hand fact on any arm.
A hand without a declaration keeps its original colours (merged URDF, vendor meshes) and the generator says so.

Values were chosen by placing the official photo (source URL) next to a render from the same angle. OmniPBR inputs
(diffuse_color_constant is linear, metallic_constant, reflection_roughness_constant).
Dependency-free module (the generator, the oracle and CPU checks load it by path).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Paint:
    diffuse: tuple[float, float, float]
    metallic: float
    roughness: float
    specular: float | None = None
    """OmniPBR specular_level (specular strength, default 0.5). None = not set (shader default)."""


@dataclass(frozen=True)
class HandLook:
    sources: tuple[str, ...]
    """Official (manufacturer) photo URLs — a reseller photo when the manufacturer site has none, noted in why."""
    rules: tuple[tuple[str, str, Paint | None], ...]
    why: str


#: Marker — the line prefix of the generator and the oracle.
TAG = "[HAND_LOOK]"
#: The arm's EE link in a rule's link regex (hand_look.rules_for replaces it).
EE_TOKEN = r"\{ee\}"

SHARPA_SHELL = Paint((0.28, 0.28, 0.32), metallic=0.6, roughness=0.45)
SHARPA_TIP = Paint((0.24, 0.24, 0.27), metallic=0.0, roughness=0.6)
DG5F_FINGER = Paint((0.36, 0.36, 0.35), metallic=0.75, roughness=0.38)
#: Matte cloth glove — roughness 0.9, specular 0.1 (default 0.5): roughness 0.6 rendered as a shiny dark grey
WUJI2_GLOVE = Paint((0.02, 0.02, 0.02), metallic=0.0, roughness=0.9, specular=0.1)

HAND_LOOKS: dict[str, HandLook] = {
    "sharpa_wave": HandLook(
        sources=(
            "https://www.sharpa.com/pages/wave",
            "https://www.sharpa.com/cdn/shop/files/Codex_2026_8_17_17_57_19.jpg",
        ),
        rules=(
            (r"\{side\}_.*|\{ee\}", r"material_CAD1EE", SHARPA_SHELL),
            (r"\{side\}_.*_DP", r"material_(3310033|1910033)", SHARPA_TIP),
        ),
        why="The official SharpaWave has a cool silver-grey matte metal shell, and the fingertips are the same grey. "
        "Without this rule the hand takes the body's white-cover rule (material_CAD1EE → bright steel 1.0) and "
        "renders near white, with the URDF's green fingertip marks.",
    ),
    "dg5f_s": HandLook(
        sources=(
            "https://www.tesollo.com/products/grippers/dg-5f-s",
            "https://cdn.tesollo.com/site/products/2026/08/146d4c2b-d994-494a-ad50-d38c48ffed34.webp",
        ),
        rules=(
            (r"\{side\}_link_\d_4", r"silicon|Material_001", None),
            (r"\{side\}_link_\d_\d", r"dsilver", DG5F_FINGER),
            (r"\{side\}_link_5_1", r"lsilver|", None),
        ),
        why="The official DG-5F-S has a white back cover (green marks), metallic titanium finger links and white "
        "fingertips. The cover, the fingertips and the thumb base (light silver) "
        "already match; only the finger links are matte dark grey (dsilver), so they get a metallic finish.",
    ),
    "wuji2": HandLook(
        sources=("https://humanoid.guide/product/wuji-hand-two/",),
        rules=(
            (r"\{side\}_.*", r".*", WUJI2_GLOVE),
            (r"\{ee\}", r"material_(left|right)_wrist_g\d+", WUJI2_GLOVE),
        ),
        why="wuji2 will wear a glove, so the whole hand is black — finger links, fingertips (including the URDF's "
        "sky blue 0.55 0.78 0.95) and palm (the wrist parts merged into the wrist link). The silver palm plate "
        "and copper parts of the official photo sit under the glove and are not followed. Arm, torso unchanged.",
    ),
}
