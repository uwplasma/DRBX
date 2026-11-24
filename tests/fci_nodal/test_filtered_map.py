"""Data-free contract checks for the opted-in filtered map."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from drbx.deck import load_deck
from drbx.fci_nodal._metric_contract import (
    BASE_MESH_SHAPE, BASE_POLOIDAL_MODES, BASE_PROJECTIONS, BASE_RADIAL_DEGREE, BASE_TOROIDAL_MODES,
    PHI_NEWTON_ITERATIONS, spline_cache_settings,
)
from drbx.fci_nodal.case._adoption import validate_adoption
from drbx.fci_nodal.deck import resolve_case_options
from drbx.fci_nodal.evaluators import cache
from drbx.fci_nodal.evaluators.eta_filtered_field import EtaFilterSpec
from drbx.fci_nodal.evaluators.exact_plane_metric import EXACT_PLANE_PAYLOAD_VERSION
from drbx.fci_nodal.evaluators.jax_metric_evaluator import JaxMetricEvaluator
from drbx.fci_nodal.evaluators.map_filter import filter_map_samples, filter_solved_map
from drbx.fci_nodal.evaluators.metric_evaluator import MetricEvaluator
from drbx.fci_nodal.options import MetricOptions

from tests.fci_nodal.test_owner_deck import write_deck


@pytest.mark.parametrize("value", ('"true"', "1", "[]"))
def test_filter_map_option_requires_boolean(tmp_path, value):
    with pytest.raises(ValueError, match="filter_map"):
        load_deck(write_deck(tmp_path, tables=f"filter_map = {value}\n"))


def test_filter_map_is_opt_in(tmp_path):
    deck = write_deck(tmp_path)
    _, config = load_deck(deck)
    default = resolve_case_options(config).metric
    deck.write_text(deck.read_text() + "filter_map = false\n")
    _, config = load_deck(deck)
    assert resolve_case_options(config).metric == default
    assert "filter_map" not in default.numerical_options()
    assert default.filtered_map_identity() == {}
    deck.write_text(deck.read_text().replace("filter_map = false", "filter_map = true"))
    _, config = load_deck(deck)
    opted_in = resolve_case_options(config).metric
    assert opted_in.numerical_options()["filter_map"] is True
    with pytest.raises(ValueError, match="requires a field filter"):
        replace(opted_in, field_filter=None)


def test_default_metric_cache_key_is_unchanged(tmp_path, monkeypatch, capsys):
    grid, wall = tmp_path/"grid.nc", tmp_path/"wall.txt"
    grid.touch()
    wall.touch()
    options = MetricOptions(makegrid_path=grid, vessel_path=wall, metric_cache_dir=tmp_path/"metric")

    def stop_before_fitting(**kwargs):
        raise RuntimeError("reached fitter")

    monkeypatch.setattr(cache, "build_metric_evaluator", stop_before_fitting)
    for current in (options, replace(options, filter_map=False), replace(options, filter_map=True)):
        with pytest.raises(RuntimeError, match="reached fitter"):
            cache.build_metric_cache(current)
        last = Path(capsys.readouterr().out.strip().split("miss: ")[1]).name
        # Content-addressed name produced from two empty inputs before the option was added.
        assert (last == "hsx_metric_98a9e1b0b2ee67e625ffe86e.npz") is not current.filter_map


@pytest.mark.parametrize("cutoff", [0, 2, 4])
def test_low_pass_on_periodic_map_channels(cutoff):
    spec = EtaFilterSpec(max_harmonic_per_period=cutoff, nfp=1, samples_per_period=64)
    eta = np.arange(spec.samples_per_period)*spec.period/spec.samples_per_period
    base = np.broadcast_to([3., .4, -.08], (eta.size, 3))
    modes = [(1, .06*np.cos(eta[:, None] + np.array([.1, .4, .7]))),
             (2, .03*np.sin(2*eta[:, None] + np.array([.2, .5, .8]))),
             (4, .01*np.sin(4*eta[:, None] + np.array([.3, .6, .9]))),
             (7, .02*np.cos(7*eta[:, None] + np.array([.1, .6, .3])))]

    def cartesian(channels):
        R, Z, delta = channels.T
        return np.stack((R*np.cos(eta+delta), R*np.sin(eta+delta), Z), axis=-1)

    position = cartesian(base + sum(sample for _, sample in modes))
    expected = cartesian(base + sum(sample for mode, sample in modes if mode <= cutoff))
    np.testing.assert_allclose(filter_map_samples(position, eta, spec), expected, atol=5e-15, rtol=0.)


@pytest.fixture
def tagged_map():
    spec = EtaFilterSpec(max_harmonic_per_period=2, nfp=4, samples_per_period=64)
    u, theta, eta = np.linspace(0, 1, 8), np.arange(8)*np.pi/4, np.arange(8)*spec.period/8
    U, T, E = np.meshgrid(u, theta, eta, indexing="ij")
    R = 3. + U*np.cos(T) + 0.03*np.cos(3*E*spec.nfp)
    X = np.stack((R*np.cos(E), R*np.sin(E), -U*np.sin(T)), axis=-1)
    solved = MetricEvaluator(u, theta, eta, X, nfp=spec.nfp, topology="toroidal",
                             poloidal_modes=1, radial_degree=3, toroidal_modes=3)
    return filter_solved_map(solved, spec)[0], spec


@pytest.mark.parametrize("loader", [MetricEvaluator, JaxMetricEvaluator])
def test_loader_dispatches_only_on_supported_tags(tagged_map, loader):
    metric, spec = tagged_map
    prefix = "map_"
    payload = metric.to_cache_payload(prefix=prefix)
    restored = loader.from_cache_payload(payload, prefix=prefix)
    q = np.array([[.7, .45, .32]])
    np.testing.assert_allclose(restored.position(q), metric.position(q), rtol=2e-14)
    assert max(map(abs, restored.toroidal_modes)) == spec.max_harmonic_per_period
    for tag in (None, "other_construction"):
        invalid = dict(payload)
        invalid.pop("map_plane_construction")
        if tag is not None:
            invalid["map_plane_construction"] = np.asarray(tag)
        with pytest.raises(ValueError, match="exact level-set planes"):
            loader.from_cache_payload(invalid, prefix=prefix)
    with pytest.raises(ValueError, match="cutoff"):
        loader.from_cache_payload(dict(payload, map_map_eta_cutoff=np.asarray(1)), prefix=prefix)


def test_adoption_matches_construction_both_ways(tmp_path, tagged_map):
    metric, spec = tagged_map
    exact = MetricOptions(makegrid_path=tmp_path/"grid.nc", field_filter=spec)
    filtered = replace(exact, filter_map=True)
    archive = tmp_path/"metric.npz"

    def validate(payload, requested, *, recorded):
        cache_spec = dict(spline_cache_settings(), resolution=[16, 16, 16], topology="toroidal",
                          metric_mesh_shape=list(BASE_MESH_SHAPE), metric_radial_degree=BASE_RADIAL_DEGREE,
                          metric_poloidal_modes=BASE_POLOIDAL_MODES, metric_toroidal_modes=BASE_TOROIDAL_MODES,
                          eta_projection_iterations=BASE_PROJECTIONS)
        cache_spec.update(recorded.filtered_map_identity())
        np.savez(archive, **payload, cache_spec=json.dumps(cache_spec), nfp=spec.nfp)
        validate_adoption("metric", {"products": {"metric_cache": archive}}, (16, 16, "nfp", spec.nfp),
                          {}, requested)

    tagged = metric.to_cache_payload(prefix="metric_evaluator_")
    validate(tagged, filtered, recorded=filtered)
    with pytest.raises(ValueError, match="construction or cutoff"):
        validate(tagged, exact, recorded=exact)
    with pytest.raises(ValueError, match="construction or cutoff"):
        validate(tagged, replace(filtered, field_filter=replace(spec, max_harmonic_per_period=1)), recorded=filtered)

    exact_markers = dict(metric_evaluator_plane_construction="exact_level_set",
                         metric_evaluator_exact_plane_payload_version=EXACT_PLANE_PAYLOAD_VERSION,
                         metric_evaluator_exact_plane_newton_iterations=PHI_NEWTON_ITERATIONS)
    validate(exact_markers, exact, recorded=exact)
    with pytest.raises(ValueError, match="construction or cutoff"):
        validate(exact_markers, filtered, recorded=filtered)
    with pytest.raises(ValueError, match="exact level-set planes"):
        validate({}, exact, recorded=exact)
