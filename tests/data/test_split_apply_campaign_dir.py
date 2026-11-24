"""split_apply makes its campaign folder: the documented call names a folder that does not exist yet
(CAMPAIGN_DIR=logs/split), and the first write (the units snapshot) used to raise FileNotFoundError."""

from __future__ import annotations

import pytest

from dexengine.data.stage1 import split_apply as SA


def test_split_apply_makes_the_campaign_folder(tmp_path, monkeypatch):
    class Stop(Exception):
        pass

    def stop(self):
        raise Stop

    camp = tmp_path / "logs" / "split"
    monkeypatch.setenv("ARM", "rby1")
    monkeypatch.setenv("CAMPAIGN_DIR", str(camp))
    monkeypatch.setattr(SA.Apply, "split_steps", stop)
    with pytest.raises(Stop):
        SA.main(["xhand_pro", "--datasets", "hot3d"])
    assert (camp / "units_before_xhand_pro.tsv").is_file()
