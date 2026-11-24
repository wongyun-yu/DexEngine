"""OakInk2 desk recovery: an object without a collision hull (its USD missing) refuses the INTAKE write instead of
fitting the desk without it."""

import pytest

from dexengine.data.support import oakink2 as O


def test_missing_hull_refuses_the_intake_write():
    res = {"clips": {}, "summary": {"no_hull_objects": ["O02@0001@00001"]}}
    with pytest.raises(SystemExit, match="without a collision hull"):
        O._refuse_partial(res, {"clips": {}})


def test_all_hulls_present_passes():
    O._refuse_partial({"clips": {}, "summary": {"no_hull_objects": []}}, {"clips": {}})
