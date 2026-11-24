"""Check external owner task dispatch without a scheduler or sockets."""

import sys
from unittest.mock import Mock

import pytest

from drbx.fci_nodal.case import _rank_launch as launcher


@pytest.mark.parametrize("options,owners,error", [
    ({"launch_mode": "scheduler"}, True, "rank_command"),
    ({"launch_mode": "scheduler", "rank_command": [""]}, True, "rank_command"),
    ({"launch_mode": "scheduler", "rank_command": "run"}, True, "rank_command"),
    ({"launch_mode": "scheduler", "rank_command": ["run", 1]}, True, "rank_command"),
    ({"launch_mode": "scheduler", "rank_command": ["run"],
      "rank_index_env": "BAD=KEY"}, True, "rank_index_env"),
    ({"launch_mode": "scheduler", "rank_command": ["run"],
      "collectives_interface": "a,b"}, True, "collectives_interface"),
    ({"launch_mode": "scheduler", "rank_command": ["run"]}, False, "require .*owners"),
    ({"launch_mode": "per_owner", "collectives_interface": "net"}, True, "require launch_mode"),
    ({"collectives_interface": "net"}, False, "require .*owners"),
])
def test_scheduler_option_errors(options, owners, error):
    with pytest.raises(ValueError, match=error):
        launcher.scheduler_options(options, owners=owners)


def test_scheduler_defaults_and_command(tmp_path, monkeypatch):
    prefix = ["job runner", "-n", "2"]
    command, key, interface = launcher.scheduler_options(
        {"launch_mode": "scheduler", "rank_command": prefix}, owners=True)
    assert (command, key, interface) == (prefix, "SLURM_PROCID", None)
    task = Mock(returncode=0)
    task.wait.return_value = task.poll.return_value = 0
    popen = Mock(return_value=task)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.socket, "gethostname", lambda: "node1")
    reservation = Mock()
    reservation.getsockname.return_value = ("node1", 1234)
    context = Mock()
    context.__enter__ = Mock(return_value=reservation)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(launcher.socket, "socket", lambda *args: context)
    env = {"GLOO_SOCKET_IFNAME": "cpu_net"}
    launcher._scheduler_team(tmp_path, [0, 1], env, 20, command, key)
    reservation.bind.assert_called_once_with(("node1", 0))
    args = popen.call_args.args[0]
    assert args == prefix + [sys.executable, "-B", "-u", "-m", launcher.__name__,
                             "--scheduler-rank", "node1:1234", "2", str(tmp_path), key]
    assert popen.call_count == 1 and popen.call_args.kwargs["env"] == env
    assert task.wait.call_args_list[0].kwargs == {"timeout": 20}
    assert sorted(p.name for p in tmp_path.glob("*.pkl")) == ["rank0000.pkl", "rank0001.pkl"]


@pytest.mark.parametrize("value,error", [(None, "decimal"), ("-1", "decimal"), ("x", "decimal"),
                                         ("2", "outside"), ("١", "decimal")])
def test_scheduler_rank_index_errors(value, error):
    env = {} if value is None else {"INDEX": value}
    with pytest.raises(ValueError, match=error):
        launcher._scheduler_rank_index(env, "INDEX", 2)


def test_scheduler_rank_host_and_duplicate_task(tmp_path, monkeypatch):
    assert launcher._scheduler_rank_index({"INDEX": "1"}, "INDEX", 2) == 1
    launcher._check_coordinator_host("node1:1234", 0, "node1")
    launcher._check_coordinator_host("node1:1234", 1, "node2")
    monkeypatch.setenv("INDEX", "0")
    monkeypatch.setattr(launcher.socket, "gethostname", lambda: "node2")
    with pytest.raises(ValueError, match="rank 0 must run on coordinator host node1"):
        launcher._scheduler_rank("node1:1234", "2", tmp_path, "INDEX")
    (tmp_path / "rank0000.log").touch()
    with pytest.raises(ValueError, match="Duplicate scheduler task for owner 0"):
        launcher._scheduler_rank("node2:1234", "2", tmp_path, "INDEX")


def test_scheduler_rejects_existing_reply(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    replies = tmp_path / "ranks"
    replies.mkdir()
    (replies / "rank0001.json").touch()
    monkeypatch.setenv("INDEX", "1")
    with pytest.raises(ValueError, match="Duplicate rank reply for owner 1"):
        launcher._scheduler_rank("node1:1234", "2", launch, "INDEX")
