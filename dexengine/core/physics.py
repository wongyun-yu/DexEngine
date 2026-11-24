"""Physical constants shared by the pipeline — each value has this one named owner.

`GRAVITY_MPS2` is the magnitude of the gravitational acceleration (m/s²) that the sim scenes run with (the Isaac /
PhysX scene default, -9.81 along world z) and that the code which reasons about weight uses: the ramped external
gravity force of the residual tasks, the contact-force direction of the contact reward, and the MuJoCo replay
scene of the carry check. Callers read it from here instead of writing the number again; the sign (down = -z) is
the caller's, at the place where a vector is built.
"""

from __future__ import annotations

#: Gravitational acceleration magnitude (m/s²). Owner of the 9.81 written in scene configs and force arithmetic.
GRAVITY_MPS2 = 9.81
