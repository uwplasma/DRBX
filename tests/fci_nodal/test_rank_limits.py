"""Data-free checks for the owner launcher policy."""

import pytest

from drbx.fci_nodal.case._rank_launch import RANK_RSS_ENV, _limits


def test_rank_limits_default_to_unlimited_and_inherit_case_guard():
    assert _limits({}, {}) == (None, None)
    assert _limits({}, {RANK_RSS_ENV: '2000'}) == (2000, None)
    env = {RANK_RSS_ENV: '2000'}
    opts = dict(rank_rss_limit_bytes=2500, run_timeout_seconds=1.5)
    assert _limits(opts, env) == (2000, 1.5)
    opts['rank_rss_limit_bytes'] = 1000
    assert _limits(opts, env) == (1000, 1.5)


@pytest.mark.parametrize('name,value', [
    ('rank_rss_limit_bytes',0),
    ('run_timeout_seconds',float('inf')),
])
def test_rank_limits_reject_nonpositive_or_unbounded_values(name,value):
    with pytest.raises(ValueError,match=name):
        _limits({name:value}, {})
