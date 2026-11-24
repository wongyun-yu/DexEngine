"""The rule list: each rule's id, title, pipeline stage, owning code and the threshold references that the
rule registry resolves from the owning files (`dexengine.core.rules.rule.resolve`).

Stage strings are matched exactly by the judge input fingerprint
(`STAGE_POSITION` of the placement judge input record); threshold references (`path:symbol`) are its keys.
"""

from __future__ import annotations

from dexengine.core.rules.cards_head import RULES_HEAD
from dexengine.core.rules.cards_run import RULES_RUN
from dexengine.core.rules.cards_tail import RULES_TAIL

#: Every rule, in declaration order.
RULES = [*RULES_HEAD, *RULES_TAIL, *RULES_RUN]
