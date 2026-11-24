"""Support thresholds shared by data filtering and the placement bake (metres unless noted).

Start gap ``g0`` of a main object that no hand holds at the start (mesh floor minus desk top, start pose):

- ``g0 < -START_EMBED_M``: the object starts embedded in the desk; the clip is excluded at the first pass (rule D24').
- ``-START_EMBED_M <= g0 < HIGH_REST_M``: the desk supports it.
- ``g0 >= HIGH_REST_M``: a disc stands under it in place of a support stand (rule D26'). The same band applies to
  rest intervals.

Support object: what holds an unheld main object up at tree frame 0 on the visual meshes -- the desk, a tracked
object under it (rack, tripod, plate, a still sibling part), a disc (nothing tracked under it), or nothing (excluded).
"""

HIGH_REST_M = 0.01
START_EMBED_M = 0.01

SUPPORT_BAND_M = 0.003  # the main object's bottom points = its vertices within this of its floor
SUPPORT_REACH_M = 0.015  # a candidate vertex counts when it is this close (horizontally) to a bottom point ...
SUPPORT_NEST_MAX_M = 0.015  # ... and no higher than the main floor + this (a rim around the bottom = nested)
SUPPORT_GAP_MAX_M = 0.010  # main floor - the highest such vertex <= this (gap in [-SUPPORT_NEST_MAX_M, +this])
SUPPORT_MIN_VERTS = 3  # that many candidate vertices under the bottom
SUPPORT_POLY_PAD_M = 0.005  # the main object's centre (xy) lies inside the contact polygon grown by this
SUPPORT_STILL_M = 0.020  # a support stays put over the tree window: origin displacement <= this ...
SUPPORT_STILL_DEG = 10.0  # ... and rotation (degrees) <= this

#: Rule id whose dataset scope (rule registry) stands pedestals under main objects in desk scenes.
PEDESTAL_RULE = "D45"
