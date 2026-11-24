"""The committed shared outputs (e.g. the articulation spec) resolve to files that exist in the repository."""

import pytest

from dexengine.core import datasets as RR
from dexengine.core.errors import InputMissingError


def test_every_shared_output_is_committed():
    for name in RR.AXIS_OUTPUTS:
        p = RR.axis_output(name)
        assert p.is_file(), p
        assert p.parent.parts[-2:] == RR.SHARED_OUTPUT_DIR


def test_unknown_shared_output_fails():
    with pytest.raises(InputMissingError):
        RR.axis_output("not_registered.json")


def test_missing_shared_output_fails(tmp_path):
    with pytest.raises(InputMissingError, match="AXIS_OUTPUT"):
        RR.axis_output(RR.ARTIC_SPEC, root=tmp_path)
