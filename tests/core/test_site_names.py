"""Deployment specifics come from the layout file: the campaign hosts and the owner-ledger root (a deployment's
DEXENGINE_LAYOUT_CONFIG), with release defaults when it names none."""

import socket
from pathlib import Path

import pytest

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.core.layout import roots
from dexengine.rl import cell_state as CH

REPO = Path(__file__).resolve().parents[2]


def _layout(monkeypatch, **kw):
    monkeypatch.setattr(roots, "_LAYOUT_CFG", [roots.LayoutConfig(**kw)])


def test_release_host_is_this_machine(monkeypatch):
    _layout(monkeypatch)
    assert CH.hosts() == {CH.LOCAL_HOST: socket.gethostname()}
    assert CH.this_host() == CH.LOCAL_HOST
    assert CH.gpu_split(f"{CH.LOCAL_HOST}:last", "", n_gpus=2) == (CH.LOCAL_HOST, ["0"], ["1"])
    with pytest.raises(SystemExit):
        CH.gpu_split("elsewhere:0", "", n_gpus=2)


def test_site_hosts_from_the_layout_file(monkeypatch):
    me = socket.gethostname()
    _layout(monkeypatch, hosts={"a": me, "b": "no-such-host"}, host_gpu_count_cmd={"b": "echo 3"})
    assert CH.this_host() == "a"
    assert CH.host_gpu_count("b") == 3
    with pytest.raises(SystemExit):
        CH.gpu_split(f"{CH.LOCAL_HOST}:0", "", n_gpus=2)  # a site table replaces the local label
    _layout(monkeypatch, hosts={"b": "no-such-host"})
    with pytest.raises(InputMissingError, match="not in the campaign hosts"):
        CH.this_host()


def test_owner_root(monkeypatch, data_root, tmp_path):
    _layout(monkeypatch)
    assert P.owner_root() == data_root
    _layout(monkeypatch, owner_root=str(tmp_path / "owner"))
    assert P.owner_root() == tmp_path / "owner"
