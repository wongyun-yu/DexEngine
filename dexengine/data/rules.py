"""Rule registry API: which data-preparation rules apply to which dataset, hand, arm and scene.

Import it as a package module, never by file path::

    from dexengine.data import rules as RR
    RR.active_hand(); RR.ledger_key("hot3d", "bimanual"); RR.rule_datasets("D45")

The implementation still lives in ``dexengine.core.datasets``. This module *is* that module object
(``sys.modules`` alias), so module state such as the active hand, the active arm and the path caches is shared,
and the lazy attributes ``ACTIVE_HAND`` / ``ARM`` resolve the same way.

Command line (same flags as the implementation)::

    python -m dexengine.data.rules --f0-axis <dataset>
"""

import sys as _sys

from dexengine.core import datasets as _impl

if __name__ == "__main__":
    raise SystemExit(_impl.main())

_sys.modules[__name__] = _impl
