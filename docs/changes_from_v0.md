# Changes from v0

v0.1 is v0 with the changes below. Each line states a change in behaviour or output; changes that leave every output
unchanged are listed last. A line marked (also in v0) is a change from the code that produced the v0 data that the v0
branch already carries.

## Data generation: scene and collision rules

- **Scene rules for every dataset:** touched still objects collide with the main objects, and a clip whose recorded
  overlap with a main object, on the shapes physics collides, is deeper than 5 mm is dropped; still objects within
  5 cm of the recorded object sweep stand in the scene; observed supports replace start discs; the render overlap gate
  also measures the main objects and extra slots; a clip whose recorded hand goes under the desk top is gated, and
  spawned objects standing off the desk are flagged.
- **Object below the desk:** a clip is dropped when the main object's lowest point goes more than 2 cm below the desk
  top in any frame, inside or outside the desk footprint, for every dataset.
- **Carried objects:** an extra object that rides another object is attached to its carrier for the whole clip.
- **Multi-clip slots:** a support disc, support object or extra object slot that holds assets of two or more clips
  always spawns per environment and keeps each environment on its clip. In v0 a one-hand run whose clips share one
  object cloned the first clip's slot into every environment, and a reset to another clip stopped on the slot check.
- **Augmentation control (pv00):** the identity variant of placement augmentation is a byte copy of the canonical
  clip's arrays (the tree's frame files except the contact files, which are remapped like every variant, and the
  arm IK and wrist caches), checked by sha256; its task_info carries the identity placement record. In v0 it
  re-solved the arm from the judge seed, so it could differ from the clip the judge passed and drop that clip (for
  example on desk penetration).
- **Placement variants and extra slots:** a placement variant (a moved and turned copy of a clip) takes the extra-slot
  classification of its source clip. The placement rule (a brushed still object in the robot keep-out or off the
  desk) stays the variant's own. A variant without its source tree beside it, with a mismatched placement record or
  with other extra objects stops, and with a ledger that carries the rule version, so does a variant whose source
  has no ledger row.
- **Receivers inside a container:** a still object receives a moved object that ends near its surface or with its
  centre inside the still object's convex hull. In v0 'inside' was the object's world-axis box, which also holds space
  outside the object and changes when the object turns.
- **Fingertip-sensor oracle:** a recording whose fingertip sensors read 0 N on every frame passes when the
  independent contact view shows force on the touched object only through links other than the fingertips
  (`ORACLE_NONTIP`). In v0 that recording stopped. A NaN anywhere in that contact view now stops the recording; in
  v0 it could pass as untouched.
- **Delivered scene:** the scene manifest states the delivered (full-physics) and the training collision tables. Each
  final carries `meta/scene_manifest.jsonl`, the scene and collision statement of every recorded clip.
- **Still-object collider:** pinned still objects collide through their visual mesh (triangles); dynamic bodies stay
  convex. In v0 they were convex cooked pieces, which made the robot appear inside empty space.
- **Robot and still objects in RL:** the robot now collides with still objects during training. Robot self-collision
  stays off.
- **Frame-0 dynamic object pairs:**
  - Two objects that keep their relative pose over the whole clip (2 mm / 2°) become one body; a pair that moves
    together without an exact one-body path is dropped.
  - Other overlapping dynamic pairs are separated at frame 0 by the frame-0 separation rule and keep their pair
    collision; only overlaps deeper than that rule allows turn the pair collision off, and such a clip is dropped if
    the visual overlap exceeds 5 mm in any frame. The pair decision reads the state training starts from: after the
    bimanual cross-repair, a pair that newly overlaps by more than 1 mm, or an off pair that the cross-repair pushed
    deeper, is read again on the final rows and the clip is dropped past 5 mm. Gate records made before this change
    are gated again once.
- **Robot self-penetration:** a clip whose output exceeds 20 mm is dropped.
- **Frame-0 separation:** a dynamic main object that starts inside a colliding body is moved out by up to 5 mm; deeper
  starts are dropped.
- **Frame-0 object repair:** frame-0 hand-object penetration deeper than 5 mm is repaired over the spawn-hold window,
  aiming at 2 mm. When no attempt reaches 2 mm, frame 0 keeps the shallowest attempt at or under 5 mm (the depth a
  side may start at without repair) that passes the repair's other checks; a side with no such attempt is dropped.
- **Frame-0 repair:**
  - The repaired wrist rotation is written into the qpos.
  - The frame-0 check places the hand with the same palm offset as training (SharpA 29.5 mm, Tesollo DG-5F-S 15 mm
    along the palm axis; wuji2 and XHand 0). In v0 the check used the hand without this offset.
  - The frame-0 hand and still-object repair runs inside the frame-0 window.
- **Floating objects:** a free main object resting above the desk or its disc settles onto it during the rest window;
  a clip whose object would jump more than 20 mm at the first touch is dropped.
- **Placement judge:** the desk check runs on the judge side instead of the stage-1 drop.
- **Earlier adoption records:** the placement judge never reads an earlier adoption record. A fresh judgement is
  unchanged, and a rerun gives the same placement as the first run.
- **Reference hand and still objects after frame 0:** measured only, never repaired.
- **Judge fix:** the all-frame confirmation mapped the coarse path to the wrong frames. This changes judge values.
- **Stage-1 step 2f (hand clear):** after the main-object contact sidecar, every base tree is measured and none is
  written. A reference hand inside a still object the robot collides with keeps its reference as recorded, with its
  depth. A clip whose true-shape object overlap, first-touch jump or moving-together pair rules it out is dropped for
  that hand. The per-(clip, side) ledger and the hand-scoped exclusion lines are read after stage 1. A tree's verdict
  is reused only while every input 2f reads is byte-equal: qpos, task_info, contact sidecars, scene files, MANO take,
  object geometry, shape rule and ruler.

## Data generation: inputs

- **HOT3D desk plane:** each HOT3D recording is levelled on its desk plane (one rotation of hands, objects and desk
  together), and the arrival distance of a receiving object is measured on surface samples of that object. An
  object is received by a still object under it when its surface comes within 5 mm of that object's top surface
  (ground-truth heights on surface samples) and its centre lies inside that object's footprint (top view). There is
  no centre-of-mass condition, so a bowl receives what lies in it. A receiver's bottom height reads the whole mesh.
- **Unit object quaternions:** the converter writes object quaternions at unit length in every clip, and every reader
  that builds a rotation requires unit length. A base built before this change is refused with its rebuild command.
- **Articulated joint values:** the joint-value sidecars of an articulated object stay on the qpos time axis through
  the frame-0 pad and the retime resample.
- **OakInk2 desk recovery:** the intake write is refused when an object has no collision hull, instead of fitting the
  desk from fewer objects.
- **Manual holds:** clips held by hand are listed in `configs/record/user_holds.tsv`, and every judging run
  reads that table; a missing table stops the run. The `--hold-clips` flag is gone.
- **Augment cell reuse:** a region dump or a judge chunk is reused only when the current judge code commit wrote it;
  anything else is judged again. Before judging, a clip whose judge inputs no longer match its adopted record is held,
  and the rest of the bundle goes on.
- **Human exclusions:** clips excluded by human decision are listed in the repository
  (`configs/datasets/<dataset>/excluded_clips.txt`). The reader takes their union with the pipeline's exclusion ledger
  in the data root, so a data root built from the raw datasets applies them too.
- **Articulated object key:** an articulated main object enters the train-set object key as `<asset id>@artic` instead
  of its mesh folder name. Its shape-family member is read from the articulated surface folder.
- **CPU bake arm IK:** an augment bake on the CPU solves the arm IK in float64, so the result does not depend on the
  host. A GPU bake keeps the float32 solver.
- **Record fields:** `retarget_diagnostics.json` names the posture prior by its path in the code tree (it was the
  checkout's absolute path). `INTAKE/h2o_table_plane.json` no longer copies `desk_z_intake`; that value stays in
  `support_audit_untilted.json`.
- **Run records:** ledgers, audits and tables of a run go to `records/` of the run tree instead of `docs/experiments/`.

## Robot and hand physics

- **Left and right:** hand friction (2.0), the mount height offset and gravity off are applied to every robot
  articulation. In v0 the left arm and hand of the split bimanual scene kept hand friction 0.5, and the mount height
  offset moved the right-side articulation only.
- **wuji2 and Tesollo DG-5F-S:** MCP hyperextension up to 20° where the vendor limit permits. On wuji2 this is all ten
  MCPs; on Tesollo only the little-finger MCP, since the other MCPs are one-sided in hardware. PIP and DIP stay cut at
  0°. The 20° applies to the simulated joint limit only; the retargeting IK keeps the 0° cut, so the kinematic
  reference is unchanged (opening it in the IK gave the same fingertip error and S-shaped fingers).
- **Tesollo finger force limit:** set to the vendor nominal fingertip force (19.4 N), the same rule as XHand and wuji2.
  v0 used the joint stall torque, which gives 8.60 N at the fingertip (0.44 of nominal). The applied joint cap becomes
  1.469 N·m (×2.26).
- **Tesollo DG-5F-S thumb retargeting:** after the IK, the thumb is re-solved kinematically with the fingertip kept
  where the IK put it, joint 1_2 inside a ±90° window and a previous-frame continuity term. On 50 TACO clips the
  Z-shaped thumb (pinned at the −153° hardware limit) goes from 42.1 % of frames to 0; thumb tip error p50/p95
  3.26/5.29 → 3.27/5.31 mm. The same window bounds the finger range of the later wrist/desk repair, the frame-0 repair
  and the finger-range gate, so no frame leaves the window after any stage. On the same clips the wrist/desk repair
  leaves 20 of 100 sides unresolved instead of 18 (two sides cross the 5 mm object-penetration limit). The simulated
  joint limit stays at the hardware value. Other Tesollo columns and the other hands are byte-identical.
- **Toothpaste flip cap:** a hinge on its tube.
- **Articulated object part masses:** split by part volume (total mass unchanged). Articulated USDs live in their own
  layer per bake generation, and their surface points live beside the USD in that layer.
- **PhysX GPU collision stack:** sized from the measured need. A buffer overflow invalidates the run.
- **Bimanual cross-repair:** the decision path uses deterministic math and the GPU pair filter drops only pairs it can
  certify; on 60 clips the outputs are byte-identical across three CPU families, with and without the GPU. Against
  v0, 2 of 50 sampled clips change repair status and wrist tracking error moves by −0.8 to +1.2 mm. The cross-repair
  process pins the OpenBLAS kernel, turns numpy's AVX-512 dispatch off and proves the loaded kernel.
- **Torso collider:** the decomposition is machine-independent (hull reduction no longer depends on the BLAS build),
  and reach maps are rebuilt. Every stage-1 entry pins the OpenBLAS kernel, turns numpy's AVX-512 dispatch off and
  proves the loaded kernel; a CPU without AVX2/FMA stops with a message.
- **Object USDs** record the convex-decomposition settings they were baked with.
- **Frame-0 input key:** the frame-0 reuse key hashes only what the frame-0 step reads (the object files it opens, the
  hand's collision links, meshes, joint chain and finger ranges) and no longer the recorded clip time. In v0 the key
  also hashed unread files (a whole mesh folder, the full robot URDF with its materials), so unrelated edits marked
  results stale while the values were unchanged. Each result also records its compute environment and the hash of each
  file it read. Frame-0 results cached by v0 are recomputed once.
- **Frame-0 articulated object cache:** the frame-0 repair keys its object geometry on the joint record as well as the
  mesh and the frame-0 joint value. In v0 two clips with the same mesh and frame-0 joint value but a different joint
  record shared one geometry inside a worker, so the result depended on solve order.
- **Generation declaration in the reuse keys:** the frame-0, judge and train-ready reuse keys now include the dataset's
  generation declaration that the contact-sidecar check reads, so a changed declaration no longer reuses results
  computed under the old one. The check's inputs are listed in one place. Keys of trees with contact sidecars change
  once; other keys are byte-identical.
- **Frame-0 wrist mapping:** the frame-0 repair writers (object window and desk) map the repaired wrist position into
  the qpos through the robot's own wrist mapping instead of one fitted from the clip. On clips whose wrists barely move
  the fitted mapping was singular, and the object-window writer failed on some wuji2 trees; these now repair.
- **SharpA frame-0 repair robot description:** the frame-0 repair reads the SharpA robot description that training and
  rendering use. v0 read an earlier merged description whose torso tilt (torso_3) differs (−0.305 ↔ −0.550). The
  repair works in palm coordinates, so the tilt cancels algebraically, but floating-point residue moves the result: on
  5 repaired SharpA clips, 2 are bit-identical and 3 differ by at most 3.7e-5 rad in joint values and 1.4e-4 in the
  wrist quaternion; the repair verdict is unchanged on all 5. The count over all SharpA trees has not been measured.

## Rendering

- **No video after bimanual training:** the bimanual training job no longer renders a video when training ends; the
  `DO_VIDEO` input and the `--no-video` flag are gone, and videos come only from the separate render step.
- **MANO render:** hand and object stay in time on retimed (padded) clips.
- **Policy videos:** `--render_markers 0` also removes the axis markers.
- **View choice record:** `orig_fov.measured` is the original clip's own grasp-visibility measurement. In v0 it was
  the last probed variant's value. The pilot check now compares the original's record with that value.
- **Floor material:** the floor no longer carries its own hidden wood material under the visible stone material.
  Trajectories and judge values are byte-identical; in the two oblique views the stone-joint detail of the floor
  differs, and the time-averaged PSNR against v0 renders is about 0.5 dB lower than between two v0 renders (front and
  ego views unchanged).

## Runtime

- **Machine-sized pools:** every worker pool takes its size from one owner that reads the machine's CPU and memory
  limits (cgroup first, then the host) and a stated memory per task, and prints it as a `[CPU] … workers=… threads=…`
  line. On a large machine the sizes equal v0's fixed counts; on a small one the stage no longer runs out of memory
  (stage-1 base layer: 8 workers were killed at 8 GiB, 1 worker completes with the same output bytes).
- **RL checkpoints:** a checkpoint is kept every 25 PPO iterations (v0: 50). Training is unchanged; there are more
  checkpoint files.
- **Dry runs and stage checks:** a dry run (`DRYRUN=1`) of a training or judging launch also starts each trainer once
  on the CPU to run the scene checks before Kit, while `DUMP_ONLY=1` writes the launch dumps only. The TACO stage-1
  count no longer expects the clips the exclusion ledger explains; `stages.py --status` and the segment to-do no longer
  stop at step 2; a PARK probe batch with trajectory recording records its cut episodes; `scripts/capped.sh` runs
  without `systemd-run` when `CAPPED_OUTER_CGROUP` names the outer memory cap; H2O can be built from the raw release
  (`dexengine.data.readers.h2o_unpack`, `read_raw h2o derive`).
- **Augmentation failures:** a failed augmentation states its cause first (`timeout`, `crash`, `no_done`, `stop`,
  `error`; a missing input keeps the `[INPUT_MISSING]` marker). The generation time cap follows the work of the clip
  (checks × frames ÷ workers, never below the old 30 min) and the bundle deadline
  (`DEXENGINE_COLLECT_BUNDLE_DEADLINE`); a child with under 60 s left is not started (cause `timeout`). Past the cap
  the whole child process group is stopped.
- **Python:** the package needs Python 3.11 or newer (v0: 3.10), as the augmentation runner starts its children with
  `subprocess` `process_group`. Both tested environments meet it (Isaac Lab 3.11, retargeting 3.12).

## Code changes with unchanged outputs

Each item below was checked by comparing syntax trees: every changed Python hunk parses to the same tree as in v0
once comments, docstrings or message text are set aside, and every other changed line is a comment.

- Formatter (ruff) layout (also in v0).
- English-only source: comments, docstrings and message text (also in v0).
- Caches keyed by source bytes (render fingerprint, stage caches) are rebuilt once, since comment and docstring edits
  change those bytes without changing behaviour.

## Other code changes

- Function-size and design checks (also in v0).
- One asset source for rendering and training: each robot (arm, hand) and object has one USD; training uses it
  (render-only prims switched off by one function) and the renderer refuses any other body (also in v0).
- Package layout `dexengine/`; hand, arm and mount are set by config (also in v0).
- Stage speed-ups (record, judge, hand-clear ruler, frame-0 separation) (also in v0).
- Robot assets keep one plain name each in paths, configs and tests (also in v0).
- The machine-sized pools change source bytes that cache keys hash: the first run after this change rebuilds the
  augment cache (generation, rollout, post gate), the judge-input parts and the torso-parts geometry cache. The 2f
  ruler's files are unchanged, so 2f verdicts stay current.
- Imports, log and error text, help text, lint fixes and code moves, with no intended output change.
- Remote job bundles state every file the scene build reads (all object surfaces, contact sidecars, MANO keypoints,
  cooked dumps, articulated parts, the H2O plane, clip lists, other-hand contact), taken from the readers' own path
  functions; an audit passes a unit only when its record shows the run opened nothing unstated. Every bundle digest
  changes, so existing bundles are repacked once.

## Known limits

- For datasets other than TACO, the reference yaw is computed from the clips present in the data root, so a root
  that holds only part of a dataset can give a different reference yaw.
- Piece clips (long clips cut into pieces) have no separate v0.1 acceptance-rate measurement.
