"""Scene requirement — the one statement of **which human-scene objects a clip's robot scene must hold**.

No imports on purpose (independent oracle): the sides read this statement and nothing else of each other —
  producer  oakink2.required (intake: chooses the raw objects to supplement into the tree)
  producer  the scene builder (which tree objects stand: extra_slots reads the stamp case.contact_frac, written by
            cases.taxonomy from hand_object_assign's contact layer over the contact sidecar)
  oracle    a separate audit (not in this package) that re-measures the raw capture and checks the tree
Each side measures on its own (frame · time base · MANO implementation · surface sampling · its own convention check);
a measurement fault on one side cannot hide on the other. Writing these values in either side again would be a second
statement — read them from here.

**A hand touched an object** has one definition for the scene (which objects stand, and the audit of it) — the scene
builder's, the six hand points (a static object the hand touched stands; touch = the builder's):
  points    the five fingertips and the palm point (palm_point: wrist + alpha · (mean fingertip − wrist))
  near      on a frame, a fingertip nearer the object surface than the fingertip threshold, or the palm point nearer
            than the palm threshold (touch_frames)
  touched   the fraction of the window's frames that are near, rounded (touch_frac), is > 0 (frac_touched) —
            no minimum time, any hand
The three numbers are not written here: each has its owner (TOUCH_TIP_REF · TOUCH_PALM_REF · TOUCH_ALPHA_REF, in the
reference grammar of data.rules) and a reader resolves the reference, never restates the value.
The hand-surface contact below (TOUCH_M · TOUCH_MIN_S · touched · SLACK_TOUCH_*) is a different measurement with its
own readers — the OakInk2 intake supplement (oakink2.required), the TACO side assignment (taco_side_assign) and the
audit's reference columns (never a defect). It does not decide which objects stand. (The intake applies the
categories below with it: before a tree exists there is no contact stamp, and what it brings in is a tree entry —
whether that entry stands is the builder's touch.)
**Grazed or handled** is a third question with its own single home — the place rule
(dexengine/placement/scene/place_rule.py,
card D48): a still extra the hand touched for less than place_rule.GRAZE_MAX_S (and that receives no main object) is
dropped from the spawn when it would stand on the robot's spot or off the desk. Its touch time is the builder's touch
above over the window (case.contact_frac × case.T / case.fps, summed over hands) — the tree holds no hand-surface
contact time, so the hand-surface distance TOUCH_M is not what that rule measures. One value, one home each:
six-point thresholds = TOUCH_*_REF · hand-surface distance = TOUCH_M · hand-surface minimum time = TOUCH_MIN_S ·
grazed/handled time = place_rule.GRAZE_MAX_S.

Per (clip window, raw object) — the first category that applies:
  moved     moved (translation >= MOVE_M or rotation >= ROT_DEG vs the window's first sample) and touched by any hand
            and no tracking jump (a centroid step >= JUMP_M)
  touched   touched by a hand — any hand: the program's activity intervals are not the pipeline's (case activity is
            the hand's own grasp time), and a hand the program calls idle holds the object it rests on
  carried   translated >= MOVE_M, touched by no hand, no jump (rides on / inside a moved object)
  receiver  not moved, and a `moved` object comes to rest on / in it: that object's surface is farther than RECV_M from
            it at the object's first window sample and within RECV_M (or its centroid inside its convex hull) at the
            last. The OakInk2 intake (oakink2.required_measure) reads 'inside' as the object's own box, which holds the
            hull: it brings in every receiver this line names, and more.
Not required: rotation-only without contact or a jump (suspect) · untouched static objects (background).

**Envelope** is a scene-builder category, not an intake category (the intake brings in only the four above, so it is
not in CATEGORIES): among the objects a tree already holds, an untouched, not-moved object that receives nothing is in
the interaction envelope when a moving object's surface (main object · moved or carried extra) comes within
ENVELOPE_M of it at some sample. Such an object stands in the scene and collides with the objects
(dexengine/data/objects/envelope.py, rule S-envelope); outside it, it stays background.
"""

# ── touch = six hand points (the scene's one definition — module doc) ────────────────────────────────────────────────
TOUCH_TIP_REF = "configs/contact/contact_source_distance.json:best_thr_m"  # fingertip threshold (m): the measured file
TOUCH_PALM_REF = "dexengine/data/cases/hand_object_assign.py:TH_PALM_M"  # palm point threshold (m)
TOUCH_ALPHA_REF = "configs/contact/mano_palm_alpha.json:alpha"  # palm point: how far from the wrist to the tips
TOUCH_FRAC_DECIMALS = 3  # the stamp case.contact_frac keeps this many decimals (hand_object_assign.assign)
# Oracle margin — the builder's thresholds are fixed, so the checker is the tighter side here: the oracle calls an
# object touched only when its own six points come nearer than the owner's threshold minus this (the judge-vs-filter
# contract from the other side — an object at a threshold must not fall between two independent measurements).
# Sized on HOT3D (2,070 clips · 12,161 measured tree objects — the oracle's raw 30 Hz posing vs the sidecar's 50 Hz
# delivered keypoints, each over the tree's own window): the two sides' minimum distances differ by 0.5 mm (median) ·
# 1.9 mm (p90), and the objects the oracle alone calls touched are 11 at no margin · 1 at 2 mm · 0 at 4 mm (the other
# way, 210 the builder alone calls touched at 4 mm — spawned, so never a defect).
SLACK_POINT_M = 0.004


def palm_point(wrist, tips, alpha):
    """The sixth point: wrist (T,3) + alpha · (mean of the five fingertips (T,5,3) − wrist). Arrays are the caller's
    (numpy semantics) — this module imports nothing."""
    return wrist + alpha * (tips.mean(1) - wrist)


def touch_frames(tip_d, palm_d, tip_m, palm_m):
    """Frame mask `near`: tip_d (T,5) fingertip ↔ surface distances · palm_d (T,) palm point ↔ surface distance ·
    tip_m · palm_m = the two thresholds (their owners: TOUCH_TIP_REF · TOUCH_PALM_REF)."""
    return (tip_d < tip_m).any(1) | (palm_d < palm_m)


def touch_frac(n_near: int, n_frames: int) -> float:
    """The fraction the builder stamps (case.contact_frac): near frames / window frames, rounded."""
    return round(n_near / n_frames, TOUCH_FRAC_DECIMALS) if n_frames else 0.0


def frac_touched(frac: float) -> bool:
    """touched = the stamped fraction > 0 (extra_slots reads the stamp this way)."""
    return frac > 0


# ── hand-surface contact (intake supplement · TACO side assignment · the audit's reference columns) ──────────────────
TOUCH_M = 0.006  # hand surface (MANO vertex) <= this from the object surface = contact
TOUCH_MIN_S = 0.10  # contact time (s) for a hand to count as touching
# ── motion · receiver (every side) ───────────────────────────────────────────────────────────────────────────────────
MOVE_M, ROT_DEG = 0.02, 5.0
JUMP_M = 0.10  # centroid step between samples = tracking jump
RECV_M = 0.015  # receiver: the mover's surface within this at its last sample (and not at its first)
CATEGORIES = ("moved", "touched", "carried", "receiver")  # first match = the order above
# Envelope margin (m): a policy that tracks the recorded objects within about 1 cm (measured RL object error of a
# trained policy: RPE 1.07 cm · MPPE 1.23 cm) stays inside 5 cm of the recorded sweep; a replay of 14 HOT3D clips
# finds 0–2 still objects per clip within it.
ENVELOPE_M = 0.05
# True-shape overlap (m): the deepest recorded overlap a pair may keep, measured on the shapes physics collides
# (collider_shape.pair_depth: a pinned still body's visual triangles, a moving body's cooked convex parts). Below it the
# reading is capture contact error: the bodies collide and separate by that much (the frame-0 object separation moves
# the dynamic object out of it). Above it the reference is dropped, never kept with the collision off: main object ↔
# touched static (S-touch-gate), mover ↔ envelope static (S-envelope), the frame-0 separation cap, and the reference
# hand inside a still extra counts as clear up to it (S-hand-clear).
TRUE_SHAPE_OVERLAP_M = 0.005
# Float settle (m): a free main object whose lowest point stands more than FLOAT_REST_M above the desk or its disc at
# the end of its rest window is set onto that support for the window (S-float-settle, data/objects/float_settle.py);
# one that would jump more than FLOAT_SETTLE_MAX_M at the first touch is dropped. FLOAT_REST_M is the census value that
# tells a float from capture noise on a resting object.
FLOAT_REST_M = 0.005
FLOAT_SETTLE_MAX_M = 0.020


def touched(touch_s: dict, min_s: float) -> bool:
    """A hand-surface contact, whatever the program says about that hand's activity: any hand's contact time (s, at
    <= the surface distance) >= min_s. min_s = TOUCH_MIN_S exactly (the audit's reference column) or loosened by the
    producer slack (the intake)."""
    return any(v >= min_s for v in touch_s.values())


# Producer slack — the producer (intake) applies every threshold loosened by these; the oracle applies them exactly.
# A producer looser than its checker keeps an object at a threshold from falling between two independent
# measurements (the judge-vs-bake-filter contract: the filter is looser than the judge). Loosened, each category
# selects a superset of its exact set; a receiver candidate is any object not already selected (a rotation just under
# ROT_DEG would otherwise leave the exact receiver set). Sized from the exact rule's misses on the full OakInk2 base
# (oakink2.required report): touch distance — both sides pose the left hand through mano_conventions;
# at 2 mm two left-hand touches the oracle saw
# at <= 6 mm stayed at 7.7 / 8.0 mm; time — one 30 Hz oracle sample plus one 50 Hz producer sample; motion — the
# median filters differ.
SLACK_TOUCH_M = 0.004
SLACK_TOUCH_S = 0.04
SLACK_MOVE_M = 0.005
SLACK_ROT_DEG = 1.0
SLACK_RECV_M = 0.005
