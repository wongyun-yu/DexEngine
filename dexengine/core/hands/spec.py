"""HandSpec: everything the training, judge and record stages need to know about one hand."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Flexion roles a hyperextension allowance is stated for (HandSpec.sim_finger_hyperext_allow_deg) — the same role names
#: and order as the retargeting prior's clip roles (prior._PRIOR_CLIP_ROLE_CHOICES; tests check they agree).
HYPEREXT_ROLES = ("mcp", "pip", "dip")


class _Unset:
    """Sentinel for a VENDOR HARDWARE FACT the porter must supply.

    Why this exists: a literal default for ``finger_effort``/``finger_velocity``
    (e.g. ``1.0``/``2.0``) would be one vendor's caps, and a new hand that
    omitted them would silently inherit those limits: no error,
    no warning, and the numbers look plausible. An inaccurate value is worse
    than a missing one: a missing one gets looked at, a wrong one never
    does. These caps feed
    the retiming gate (`dexengine/placement/retime/warp.py`), so a wrong value makes the whole
    reference chain valid-looking against the wrong hardware.

    They cannot simply drop their default: dataclass forbids a non-default field
    after defaulted ones (``finger_kp`` etc. legitimately default to ``None``).
    So the default is this sentinel and ``__post_init__`` fails loudly instead.

    Finger GAINS (``finger_kp``/``finger_kd``) are required too, but they are
    checked separately (``_REQUIRED_GAIN_FIELDS``): they default to ``None`` so
    that dataclass ordering works, and ``__post_init__`` refuses ``None``. There
    is no shared finger gain any more — every vendor cap differs, so a shared
    Kp saturates the PD on most hands. ``finger_armature`` stays a
    shared default (``None`` = ``dexengine.rl.tasks.cfg.constants.FINGER_ARMATURE``).
    """

    __slots__ = ()

    def __repr__(self) -> str:  # shows up in the error message
        return "<must be supplied from the vendor URDF>"


_UNSET = _Unset()


#: Finger PD cap rule — the error (rad) at which the PD command reaches the vendor effort cap, and Kd / Kp.
#: A 0.2 rad (~11 deg) error commands the cap, so typical tracking errors stay well inside it (no bang-bang);
#: Kd = 0.25 Kp. The rule the wuji / XHand tables were derived from (rounded there) and SharpA's retune started from.
FINGER_CAP_RULE_ERR_RAD = 0.2
FINGER_CAP_RULE_KD_RATIO = 0.25

#: The wrist share of one hand's policy action (position 3 + rotation 3). One hand's action = ``[0:6]`` wrist ·
#: ``[6:6+n]`` fingers — the layout the residual action term and the finger-penalty recipe use.
WRIST_ACTION_DIMS = 6


def finger_gains_from_cap(effort):
    """(Kp, Kd) from a vendor effort cap by the cap rule: Kp = cap / FINGER_CAP_RULE_ERR_RAD,
    Kd = FINGER_CAP_RULE_KD_RATIO · Kp. ``effort`` is a scalar (N·m) or a per-joint regex dict (same keys
    returned) — the same shape ``finger_effort`` takes. A non-positive cap raises (no silent zero gain)."""

    def _one(cap):
        cap = float(cap)
        if not cap > 0.0:
            raise ValueError(f"finger_gains_from_cap: effort cap {cap!r} must be > 0")
        kp = cap / FINGER_CAP_RULE_ERR_RAD
        return kp, FINGER_CAP_RULE_KD_RATIO * kp

    if isinstance(effort, dict):
        pairs = {k: _one(v) for k, v in effort.items()}
        return {k: v[0] for k, v in pairs.items()}, {k: v[1] for k, v in pairs.items()}
    return _one(effort)


@dataclass(frozen=True)
class HandSpec:
    """Static description of one right-hand end-effector variant."""

    name: str
    """Identifier, e.g. ``"wuji"`` / ``"sharpa_wave"``."""

    finger_joint_names_right: tuple[str, ...]
    """Independent finger joints, ACTION order (must match the retarget dataset
    qpos order after the 6 virtual-wrist dims for fully-actuated hands)."""

    coupled_joint_names_right: tuple[str, ...]
    """Mimic-coupled joints (empty for fully-actuated hands)."""

    coupling_ratios: tuple[float, ...]
    """Per-coupled-joint ratio (chained ratios pre-multiplied)."""

    coupling_source_idx: tuple[int, ...]
    """Index into finger_joint_names_right of the driver per coupled joint."""

    hand_qpos_dim: int
    """Per-hand qpos width in the retargeted trajectory NPZ (6 + n_fingers
    for fully-actuated hands; 18 for RH56F1's 6+12 layout)."""

    tip_body_names_right: tuple[str, ...]
    """5 fingertip bodies (thumb..pinky) carrying the contact pad."""

    palm_fk_body_right: str
    """Palm body name — the name on the **retargeting FK export (MJCF)** axis."""

    """FK/palm base body name in the rig model (npz body_positions)."""

    contact_body_regex_right: str
    """Regex (relative to the Robot prim) matching ALL right-hand bodies for
    the multi-body contact sensor."""

    # The EE → wrist offset is not a hand fact: it lives in the (arm, hand) mount (core.mounts.MOUNT_REGISTRY), made
    # from the arm's flange, the mount's adapter and this hand's mount_face_to_root. The full-robot USD path is a pair
    # artefact too (paths.combo_usd_relpath), so neither is a field here.

    urdf_subtree_root: str
    """Root link of the hand subtree the full-robot merge cuts from the vendor document, with ``{side}`` for the side
    (before the side-prefix rename of a vendor without one): the child of the mount joint."""

    mount_face_to_root: float
    """Distance (m) from the hand's mounting face to ``urdf_subtree_root`` along the root's +z — the hand's part of
    the mount stack (``MountSpec`` ``translation.stack``). DG-5F-S: its vendor ``joint_base`` 0.015 (the vendor mount
    plate ``link_mount`` sits below the cut root ``link_base``); a hand whose root origin is the mounting face: 0."""

    urdf_palm_frame_right: str = ""
    """Palm link name — the name on the **URDF/USD** axis. It may differ from the FK axis (RH56F1:
    FK 'right_hand_base' vs URDF 'right_hand_base_link'). This is the EE frame arm IK and placement
    optimization solve for with Pinocchio; without it here, every caller would re-write a per-hand
    name branch."""

    palm_frame_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """rig palm body (palm_fk_body) → URDF palm frame (urdf_palm_frame) translation, in **palm-local frame**
    coordinates. When the two frames are not the same physical point, the IK/placement solvers must shift the wrist
    target by this value for the fingers to overlap the retargeting FK (green dots):
    p_target' = p_rig + R_rig @ off. Same value on both sides (measured left/right Δ agree).

    Example (SharpA): the rig's C_MC is 2.95 cm further toward the fingertips than URDF C_MC — a pure translation
    Δ=(0,0,+0.0295) on all four finger MCPs and both sides, axes aligned.
    A non-zero value must be measured by a multi-child, both-side translation comparison."""

    palm_site_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation that moves the `{side}_palm` **site** (rig MJCF only) away from the `palm_fk_body` origin, in
    **right palm_fk_body local** coordinates. Default 0 = the root origin.

    **Convention — what this site is.** Retargeting IK welds this site to the **MANO wrist (joint 0)**: the target is
    `qpos_wrist_{side}` of the mano layer's `trajectory_keypoints.npz`, whose frame is Z = wrist→middle MCP ·
    Y_aux = ring MCP→index MCP (the palm weld of the hand tracker).
    So the site belongs at **the point that plays MANO joint 0's role on that hand** — neither the vendor mounting
    flange (ISO mount) nor the rig root origin MuJoCo leaves is that point.
    `palm_fk_body_right` is a **different axis** (the FK body name body_positions.npz, the wrist cache and the mount
    use). Fingertips are held strongly (connect) and the palm weakly (weld), so a misplaced site shows up directly as
    the **systematic part of the palm error**.

    **Procedure (the value is this procedure's output — not a hand-tuned number).**
    The palm site registration tool (`<hand> --clips <canonical 50>`) solves
    `s* = argmin_s Σ_i |R_i s + t_i − wrist_target_i|²  =  −mean_i(R_i^T(site_xpos_i − target_i))`.
    The left rig is the right one's Y mirror, so left measurements are averaged with Y flipped.
    The rig builder (`dexengine.robots.rig.sites`) applies the left side with Y flipped too; before that it
    **measures** that the left and right vendor URDFs really are Y mirrors and dies if not (a non-mirrored hand cannot
    use this field).

    Example (DG-5F-S): on 25 of the canonical 50 clips (4,391 frames), right (7.98, 5.20, 8.05) · left mirrored
    (9.20, 4.03, 8.82) mm → mean (8.6, 4.6, 8.4) mm. The palm error p50 was 18.5/17.3 mm, of which 12.5/13.4 mm was
    this systematic part."""

    palm_site_mano_frame: tuple[str, str] = ()
    """Source of the `{side}_palm` site's **orientation** = (robot document path (relative to the data root), right
    MANO wrist site link). Empty: the site keeps the axes of `palm_fk_body`.

    **Why.** IK welds this site to the MANO wrist target **site-to-site**, enforcing orientation too — the site axes
    declare 'where the MANO wrist axes are on this hand'. Other hands' vendor root axes happen to match the rig
    convention (fingers +z · palm normal +x), but a hand whose vendor root axes differ attaches flipped if left as is
    (wuji2: 180°, a claw whose fully curled fingers never reach the fingertips — palm error 12 cm).

    **We do not measure the value.** Used only when the robot document puts the hand root and the MANO wrist site as
    **fixed children of the same parent link**; the rig builder reads their relative rotation from those two fixed
    joints and converts the rrc MANO axis convention → the rig convention (convention owner =
    `dexengine.robots.rig.sites.RRC_MANO_AXES`). A different parent raises."""

    joint_pos_defaults: dict[str, float] = field(default_factory=dict)
    """RIGHT-side joint -> initial joint value. Empty: every joint of the hand starts at 0.

    Why: "all joints at 0" is not universally valid. A joint's limits may not contain 0 (seen on wuji's thumb:
    [0.0475, 1.603]), and Isaac Lab then refuses with ``default positions out of the limits`` (the combo env build
    fails). Hands without such a joint leave this empty.

    Left keys are generated by ``mirror_right_to_left``, so list the right side only.
    """

    vendor_link_prefix: str = "{side}_"
    """Side prefix the vendor URDF uses on link and joint names. `{side}` is replaced by right/left.

    Measured: in wuji's vendor URDF 26 of 27 links start with `right_`; in DG-5F-S **none** do (a single-hand URDF,
    the file decides the side). The converter looks up HandSpec names (`right_…`) in the URDF, so without this it dies
    looking for `right_link_base`. Declared per hand, never guessed — a silent fallback would pick up another hand.
    """

    display_name: str = ""
    """Human-readable hand name (page titles · reports) — e.g. "Tesollo DG-5F-S". Not the hand id (`name`).
    This field alone owns it — the page generator dies when it is empty."""

    vendor_urdf: str = ""
    """Vendor URDF path, relative to the repo root, with ``{side}`` for the side.

    Why it lives here: ``dexengine/robots/hand_rig.py`` — the one converter that turns a vendor URDF into a PATH A
    rig MJCF — reads its ``--hand`` choices and asset locations from the specs, so registering a hand is one place and
    the asset location stays with the rest of the hand's facts.

    **It is the document the robot is built from** — the one the Isaac merge (`dexengine.robots.urdf.merge`) and the
    MJCF rig were made from — so a check that measures it measures the robot (wuji's used to
    name `wujihand_{side}_retarget.urdf`, a different document whose left `link4` frame points −z, while the rig
    and the USD come from `model_v1.2.1_wuji.urdf`; the fingertip check then failed on geometry no robot has).
    A two-hand document carries no ``{side}`` (``format`` leaves it as is); the side then lives in ``vendor_meshes``.

    Empty means the hand has no vendor URDF, and every consumer says so instead of guessing. Vendor files are not in
    the repository: ``python -m dexengine.robots.vendor fetch`` places them (``configs/robots/vendor/``).
    """

    vendor_meshes: str = ""
    """Mesh directory for ``vendor_urdf``, same ``{side}`` convention. "" = the URDF's directory."""

    merge_source: dict[str, str] = field(default_factory=dict)
    """Side → the document the full-robot merge cuts that side's subtree from, when it is not ``vendor_urdf``
    (repository-relative, the same place rule). Read only through ``merge_document(side)``."""

    vendor_mjcf: str = ""
    """Vendor MJCF path (``{side}``, the same place rule as ``vendor_urdf``) when the retargeting rig is built from
    the vendor's MJCF instead of its URDF (``dexengine/robots/rig/from_mjcf.py``). "" = the rig is built from
    ``vendor_urdf``. The robot body (merge, USD) is always built from ``vendor_urdf``."""

    rig_palm_site: tuple[float, float, float] = ()
    """Palm site position of a ``vendor_mjcf`` rig, in its root body frame (metres). Required with ``vendor_mjcf``."""

    rig_capsule_distal_start: float = 0.0
    """Distance along the tip body's +x where the distal collision capsule of a ``vendor_mjcf`` rig starts."""

    rig_capsule_middle: bool = False
    """A ``vendor_mjcf`` rig also gets a middle-segment capsule on each tip body's parent (origin → tip body origin)."""

    retarget_dirname: str = ""
    """``processed/<dataset>/`` retarget directory name; "" = ``name``.

    When a hand uses a directory that differs from its name, **this one place** owns that mapping —
    combo_data_paths (env cfg keying) and the batch/IK runners read it here and never re-write it.
    """

    abduction_joint_regex: str = ""
    """fullmatch regex of the finger **abduction (AA) joints** (side-independent — starts with `.*`). "" = none
    declared.

    `finger_rom_floor` (the ROM lower-bound floor) must exclude these joints. A wuji joint-name literal as the action
    cfg default matched 0 joints on other hands and silently cut the abduction joints' negative lower bound, so the
    set is a fact of the hand and this field owns it. With the floor on and this empty, the action contract
    (dexengine/rl/tasks/mdp/actions/contract.py) dies."""

    finger_kp: dict[str, float] | float | None = None
    finger_kd: dict[str, float] | float | None = None
    finger_armature: float | None = None
    """Finger PD gains and rotor inertia.

    ``finger_kp`` / ``finger_kd`` are **REQUIRED per hand** (``__post_init__`` refuses ``None``): the
    gain must match that hand's vendor effort cap, or the PD saturates at a fraction of a milliradian and
    runs bang-bang. The default rule is ``finger_gains_from_cap(finger_effort)`` (Kp = cap / 0.2 rad,
    Kd = 0.25·Kp — a 0.2 rad error commands the cap). A hand may carry a tuned table instead (SharpA's
    per-class retune; wuji/XHand's rounded cap-rule tables) — say why next to it. There is no shared
    finger gain to fall back to.

    ``finger_armature`` is still control policy: ``None`` = the shared
    ``dexengine.rl.tasks.cfg.constants.FINGER_ARMATURE``; a scalar equal to it is a COPY."""
    finger_effort: dict[str, float] | float = _UNSET  # type: ignore[assignment]
    """Effort limit (Nm). Dict = per-joint regex map (vendor URDF caps).
    **REQUIRED** — read it off the vendor URDF; see ``_Unset``."""
    finger_velocity: dict[str, float] | float = _UNSET  # type: ignore[assignment]
    """Velocity limit (rad/s). Dict = per-joint regex map.
    **REQUIRED** — read it off the vendor URDF; see ``_Unset``."""
    finger_effort_peak_scale: float = 1.0
    """Applied actuator cap = ``finger_effort`` × this (``finger_effort_applied``). 1.0 = the spec's effort IS
    the hardware maximum (SharpA — no published fingertip force). > 1 when the vendor's published fingertip force is
    above what the spec's effort gives on the rig (wuji2 · XHand · Tesollo): the scale is set so the sim's best
    index-finger distal-pad force equals ``vendor_fingertip_force_n`` — never above it."""
    vendor_fingertip_force_n: float | None = None
    """Vendor-published maximum fingertip force (N) — the ceiling the oracle holds the applied caps under.
    ``None`` = not published (the scale must then stay 1.0)."""
    finger_velocity_peak_scale: float = 1.0
    """Applied joint-speed limit = ``finger_velocity`` × this (``finger_velocity_applied``).
    1.0 = the spec's velocity IS the hardware maximum. > 1 only when a vendor-published speed figure shows the URDF
    velocity is below the hardware:
    ``vendor_close_repetition_hz`` (open-close cycles per second) needs at least 2·A·f of joint speed for a close of
    amplitude A, so a URDF speed that only allows a tiny amplitude at that frequency is not the hardware limit. The
    applied speed must stay at or below a full-range cycle at that frequency (2·range·f)."""
    vendor_close_repetition_hz: float | None = None
    """Vendor-published close (open-close) repetition frequency (Hz) — the evidence for ``finger_velocity_peak_scale``.
    ``None`` = not published (the scale must then stay 1.0)."""
    finger_velocity_peak_scale_overrides: dict[str, float] = field(default_factory=dict)
    """Per-key scale for a ``finger_velocity`` dict entry (same key string) where the scalar scale would exceed a
    full-range open-close at ``vendor_close_repetition_hz`` (2·range·f) — a short-range joint. Keys must exist."""
    vendor_urdf_limit_placeholder: dict[str, float] = field(default_factory=dict)
    """{``finger_effort`` | ``finger_velocity``: the vendor URDF ``<limit>`` value that is a PLACEHOLDER} — the
    named exception to "the spec equals the vendor URDF". The manual wins: this spec's
    value is what the sim applies, and the URDF figure is not evidence for it. A finger joint whose spec value
    differs from the URDF is allowed only when its URDF value is the one named here. The exception expires by
    structure — the vendor fixing the value, or the spec coming to equal the URDF (a placeholder no joint uses).
    A key outside ``_REQUIRED_HARDWARE_FIELDS`` raises.
    The manual source goes next to the value."""

    fingertip_site_offsets: dict[str, float | tuple[float, float, float]] = field(default_factory=dict)
    """Where the physical pad sits inside each tip BODY, in that body's local
    frame, metres. Leave empty when the tip body's ORIGIN already is the pad --
    that is the RH56F1 case and yields zero offsets.

    Two key shapes: ``{"thumb": d, "fingers": d}`` when the thumb differs and the
    rest share a value (SharpA), or one key per name in ``tip_body_names_right``.
    A value may be a scalar (metres along tip-local +x, the SharpA convention) or
    an explicit 3-vector when the vendor's tip frame puts the pad elsewhere.

    This IS the live source: ``dexengine/rl/tasks/mdp/naming.resolve_fingertip_site_offsets``
    reads it off whichever spec the asset carries (no hand has offsets baked into the resolver).

    **Left hand.** Values are right-body local, and the left hand uses **the same values in its own body-local
    frame** — correct only for hands whose vendor puts the left fingertip frames in the same orientation as the
    right (DG-5F-S · SharpA). On XHand the thumb distal link's pad axis is +y on the right and **−y** on the left, so
    reusing the value puts the left thumb site 62.5 mm off the pad (silently wrong). Such hands state that side's value
    under the **left body name key** (`mirror_right_to_left(tip)`); without one the right value is used. The one
    reader is `fingertip_site_offsets_for(spec, side)`. Whether a left key is needed is not guessed —
    `dexengine.robots.rig.sites.check_tip_sites` measures the **pad** on both sides' vendor URDF colliders
    (far-end gap · lateral gap ≤ `fingertip_pad_lateral_max_m`) and dies unless **left and right measurements agree**.
    A left key's value is measured on that side's vendor URDF pad — never guessed as the mirror of the right value
    (y·z sign flip; on wuji's real USD that mirror is 48–66 mm off)."""

    fingertip_pad_lateral_max_m: float | None = None
    """How far (m) a fingertip site may sit sideways from the pad cap centre — **declared per hand, no default**
    (None makes `dexengine.robots.rig.sites.check_tip_sites` die). Training and judge do not read it (check only).

    Why per hand: in the real build geometry the lateral gap is wuji 4.1–5.8 · SharpA thumb 4.8 mm (all toward the
    palm — the vendor sites sit on the palm-side face, not the cap centre), while the DG-5F-S mirror sign error this
    rule targets is 4.98–5.00 mm, so one global threshold cannot separate them. A direction rule such as 'allowed
    when toward the palm' cannot either — the DG-5F-S mirror error itself is +5.0 mm toward the palm (flexion sign
    taken from the parallel asymmetric joint of the same finger · thumb and pinky sign undetermined).
    So the mirror error is caught by the **left/right measurement match** (a structural rule, check_tip_sites), and
    this value is the absolute cap against both sides being wrong together. Value = the current lateral max measured
    on that hand's vendor URDF + 1 mm, rounded up to 0.5 mm (measured values in each hand's comments)."""

    sim_finger_limits: str = "vendor"
    """The **kind** of finger joint limits the sim (training · RL eval · render) spawns — "vendor" (the vendor URDF
    as is · default) | "truncated" (a copy with only the hand joint `<limit>` cut by the retargeting hyperextension
    rule prior.hyperext_clip_range — generator dexengine/robots/urdf/truncate.py · the range clip selection references
    already follow). Kind → asset suffix is owned by dexengine.core.arms.HAND_LIMIT_SUFFIX; the one reader is
    dexengine.core.arms.sim_hand_limits (plate = plate_variant(arm, hand)).
    Judge · IK · bake (the offline axis, canon_variant) never read it."""

    sim_finger_hyperext_allow_deg: dict[str, float] | None = None
    """How far (degrees) a flexion joint of the **sim body** may bend past its neutral angle backwards
    (hyperextension) — per flexion role of the retargeting prior chain, all three stated: ``{mcp: …, pip: …, dip: …}``;
    ``null`` = 0 for every role (the sim cut stays at neutral). Only the truncated sim asset reads it (truncate.py,
    through the cut window's ``--sim``: cut = max(vendor lower, −allowance) for sign +1) — training, RL eval and render
    spawn that body. The retargeting IK and the reference finger range (f0 · placement repairs and gate) never read
    it: they keep the cut at neutral (prior.IK_HYPEREXT_ALLOW_RAD), so stage-1 outputs and identities do not move.
    One reader: hands.helpers.sim_hyperext_allow_deg. A non-zero role the prior does not cut dies (no consumer)."""

    # Vendor hardware facts that MUST be branched per hand. Sharing these across
    # hands is never correct — they come off each vendor's URDF. Listed here (not
    # buried in __post_init__) so a porter can see the contract at a glance.
    _REQUIRED_HARDWARE_FIELDS = ("finger_effort", "finger_velocity")
    # Finger gains every hand must carry (no shared default — see the field docstring).
    _REQUIRED_GAIN_FIELDS = ("finger_kp", "finger_kd")

    def __post_init__(self) -> None:
        """Fail loudly when a vendor hardware fact was not branched.

        The rule: **hand-specific values must always branch explicitly; everything else shares the default.**
        Silence is the danger — a new hand that omitted `finger_effort` would inherit RH56F1's 1.0 Nm and nothing
        would say so.
        """
        missing = [f for f in self._REQUIRED_HARDWARE_FIELDS if isinstance(getattr(self, f), _Unset)]
        if missing:
            raise ValueError(
                f"HandSpec({self.name!r}): {', '.join(missing)} not set. "
                "These are VENDOR HARDWARE facts and must be branched per hand — "
                "read them off that hand's URDF (effort= / velocity= on each finger "
                "joint). Do NOT copy RH56F1's values.\n"
                "  Gains are separate: finger_kp/finger_kd are required too "
                "(finger_gains_from_cap(finger_effort) is the rule)."
            )
        no_gain = [f for f in self._REQUIRED_GAIN_FIELDS if getattr(self, f) is None]
        if no_gain:
            raise ValueError(
                f"HandSpec({self.name!r}): {', '.join(no_gain)} not set. Every hand carries its own "
                "finger PD gains — there is no shared finger gain to fall back to (a shared Kp saturates "
                "a small vendor cap at a fraction of a milliradian). Use the cap rule "
                "`finger_gains_from_cap(<finger_effort>)` (Kp = cap/0.2 rad, Kd = 0.25·Kp) or a tuned "
                "table with the reason next to it."
            )
        if not self.finger_effort_peak_scale >= 1.0:
            raise ValueError(
                f"HandSpec({self.name!r}): finger_effort_peak_scale {self.finger_effort_peak_scale!r} "
                "< 1 — the scale lifts a rated URDF figure to the hardware peak, it never lowers it"
            )
        if not self.finger_velocity_peak_scale >= 1.0:
            raise ValueError(
                f"HandSpec({self.name!r}): finger_velocity_peak_scale {self.finger_velocity_peak_scale!r} < 1"
            )
        if self.finger_velocity_peak_scale != 1.0 and self.vendor_close_repetition_hz is None:
            raise ValueError(
                f"HandSpec({self.name!r}): finger_velocity_peak_scale {self.finger_velocity_peak_scale} "
                "needs vendor_close_repetition_hz — a speed scale with no published figure is "
                "not hardware"
            )
        if self.finger_effort_peak_scale != 1.0 and self.vendor_fingertip_force_n is None:
            raise ValueError(
                f"HandSpec({self.name!r}): finger_effort_peak_scale {self.finger_effort_peak_scale} "
                "needs vendor_fingertip_force_n — a scale with no published ceiling is not "
                "hardware"
            )
        unknown = sorted(set(self.vendor_urdf_limit_placeholder) - set(self._REQUIRED_HARDWARE_FIELDS))
        if unknown:
            raise ValueError(
                f"HandSpec({self.name!r}): vendor_urdf_limit_placeholder keys {unknown} — only "
                f"{self._REQUIRED_HARDWARE_FIELDS} have a vendor URDF <limit> to stand in for"
            )
        if set(self.merge_source) - {"right", "left"}:
            raise ValueError(f"HandSpec({self.name!r}): merge_source keys {sorted(self.merge_source)} — right|left")
        if bool(self.vendor_mjcf) != (len(self.rig_palm_site) == 3):
            raise ValueError(
                f"HandSpec({self.name!r}): vendor_mjcf and rig_palm_site go together (a vendor_mjcf rig states its "
                f"palm site; a URDF rig derives it from palm_site_offset)"
            )
        allow = self.sim_finger_hyperext_allow_deg
        if allow is not None:
            if set(allow) != set(HYPEREXT_ROLES):
                raise ValueError(
                    f"HandSpec({self.name!r}): sim_finger_hyperext_allow_deg keys {sorted(allow)} — state every role "
                    f"{list(HYPEREXT_ROLES)} (0 where the cut stays at neutral; no silent default)"
                )
            bad = {k: v for k, v in allow.items() if not (math.isfinite(v) and 0.0 <= v < 90.0)}
            if bad:
                raise ValueError(f"HandSpec({self.name!r}): sim_finger_hyperext_allow_deg {bad} — need 0 ≤ deg < 90")

    def merge_document(self, side: str) -> str:
        """Repository-relative document the full-robot merge cuts the ``side`` subtree from."""
        return self.merge_source.get(side) or self.vendor_urdf.format(side=side)

    @property
    def n_fingers(self) -> int:
        return len(self.finger_joint_names_right)

    def left(self, names: tuple[str, ...] | None = None) -> tuple[str, ...]:
        """Mirror a right-hand name tuple to the left side."""
        src = names if names is not None else self.finger_joint_names_right
        return tuple(n.replace("right_", "left_", 1) for n in src)


@dataclass(frozen=True)
class HandRLRecipe:
    """RL training settings of one hand (the hand spec file's ``rl_recipe``). One field = one launch channel
    (``hands.recipes.RECIPE_CHANNELS``).

    A value equal to the channel's code default (scale 1.0, None) is not passed, so turning the recipe on leaves
    the argv/env of the canonical command byte-identical.
    """

    #: env DEXENGINE_FINGER_EFFORT_SCALE — scale multiplied on top of the applied cap (finger_effort_applied).
    finger_effort_scale: float = 1.0
    #: --finger_effort — build cap (N·m, every finger joint). The starting cap of the effort curriculum.
    finger_effort: float | None = None
    #: --finger_effort_anneal_start — training progress at which the build cap → hardware cap ramp starts.
    finger_effort_anneal_start: float | None = None
    #: --imit_release_min — minimum scale on the finger imitation reward in reference hand-contact frames.
    imit_release_min: float | None = None
    #: --imit_release_start — training progress at which the imitation-release ramp starts.
    imit_release_start: float | None = None
