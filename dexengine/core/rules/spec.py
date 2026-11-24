"""The `Rule` record of the rule registry."""

from __future__ import annotations

from dataclasses import dataclass, field

from dexengine.core.rules.profiles import SCENES


@dataclass
class Rule:
    """One pipeline rule.

    id          short id (`D24`, `J-desk-pen`, ...)
    title       one-line description
    stage       pipeline stage tokens joined by `·`; the judge input fingerprint maps each token to a stage
                (`dexengine.placement.judge.input_record.STAGE_POSITION`), so tokens are matched exactly
    owner       the code that implements the rule, `path:symbol`
    datasets    datasets the rule applies to; empty = all datasets
    scope_from  where the dataset scope is read instead (`path:symbol` of a per-dataset dict, or
                `profile:<field>=<value>`); `resolve()` fills `datasets` from it
    thresholds  ((label, `path:symbol`), ...) — values are read from the owning file, never copied here
    resolved    {label: value}, filled by `resolve()`
    scenes      scenes whose clips the rule applies to (default: every scene)
    """

    id: str
    title: str
    stage: str
    owner: str
    datasets: tuple = ()
    scope_from: str = ""
    thresholds: tuple = ()
    resolved: dict = field(default_factory=dict)
    scenes: tuple = SCENES
