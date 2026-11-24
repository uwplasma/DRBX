"""Exercise the deck entry points with receipt-only case stages."""

from dataclasses import asdict

import pytest

from drbx.cli import main
from drbx.deck import load_deck
from drbx.fci_nodal import case
from drbx.fci_nodal import deck as nodal_deck
from drbx.fci_nodal.case._receipts import PREDECESSORS, code_files, stage
from drbx.fci_nodal.case.case import _settings


def write_deck(tmp_path, owner_setting="", tables=""):
    deck = tmp_path / "case.toml"
    deck.write_text(
        f"""
[model]
backend = "fci_nodal"
[fci_nodal]
case_dir = "case"
resolution = 16
symmetry = "nfp"
field_periods = 4
{owner_setting}
[geometry]
makegrid_path = "grid.nc"
vessel_path = "vessel.txt"
makegrid_currents = [1.0]
rings = 9
resolution = [16, 16, 16]
[field_filter]
max_harmonic_per_period = 3
samples_per_period = 64
table_grid = [257, 256, 1.02]
"""
        + tables,
        encoding="utf-8",
    )
    (tmp_path / "grid.nc").touch()
    (tmp_path / "vessel.txt").touch()
    return deck


@pytest.mark.parametrize(
    "owner_setting, tables, parts, launch, rss, timeout",
    [
        ("", "", 1, "local", None, None),
        ("", "[parallel]\nworkers = 2", 1, "local", None, None),
        ("threads = 4", "", 1, "local", None, None),
        ("owners = 1", "", 1, "local", None, None),
        ("owners = 2", "", 2, "local", None, None),
        (
            "owners = 2", '[mms]\nlaunch_mode = "per_owner"\n'
            "rank_rss_limit_bytes = 1073741824\nrun_timeout_seconds = 120.5",
            2, "per_owner", 1073741824, 120.5,
        ),
        (
            "owners = 2", '[mms]\nlaunch_mode = "scheduler"\n'
            'rank_command = ["task-runner", "-n", "2"]\n'
            'rank_index_env = "TASK_INDEX"\ncollectives_interface = "cpu_net"\n'
            'rank_rss_limit_bytes = 1073741824\nrun_timeout_seconds = 120.5',
            2, "scheduler", 1073741824, 120.5,
        ),
    ],
    ids=("default-owner", "q-workers-are-independent", "different-build-and-run-threads", "one-owner",
         "two-local-owners", "one-process-per-owner", "scheduler"),
)
def test_owner_count_reaches_case_driver(tmp_path, monkeypatch, owner_setting, tables, parts,
                                       launch, rss, timeout):
    deck = write_deck(tmp_path, owner_setting, tables)
    calls = {}

    def build_case(options, directory):
        calls["build"] = options
        assert directory == tmp_path / "case"
        stage_threads = options.stage_threads
        receipts = {}
        stage_options = {
            name: asdict(getattr(options, attr))
            for name, attr in (("metric", "metric"), ("polygon", "polygon"), ("geometry", "geometry"),
                               ("parallel_operators", "parallel"))
        }

        def record(name, options, products=None):
            products = {} if products is None else products
            receipts[name] = stage(
                directory, name, settings=_settings(name, stage_threads=stage_threads),
                options=options, inputs=(), expected=products,
                dependencies=[receipts[item] for item in PREDECESSORS[name]],
                action=lambda out: (products, 0, 0, code_files(name)),
            )

        for name in PREDECESSORS:
            record(name, stage_options.get(name, {}))
            if name == "perpendicular_operators":
                record("owner_inputs", {"owner_count": parts, "layout": [16, 16, "nfp", 4],
                                        "build_receipts": [stamp for _, stamp in receipts.values()]})
                return directory / "owner_inputs/receipt.json"

    def run_mms(directory, options, *, stage_threads):
        calls["run"] = options
        calls["run_threads"] = stage_threads
        assert directory == tmp_path / "case"
        return {"steps": 3, "time": 0.001}

    monkeypatch.setattr(case, "build_case", build_case)
    monkeypatch.setattr(case, "run_mms", run_mms)
    assert main(["build", str(deck)]) == 0
    publication = tmp_path / "case/owner_inputs/receipt.json"
    built_receipt = publication.read_bytes()
    build_threads = 4 if owner_setting == "threads = 4" else 2
    if build_threads == 4:
        deck.write_text(deck.read_text().replace("threads = 4", "threads = 8"))
    assert main(["build", str(deck), "--verify-only"]) == 0
    assert main(["run", str(deck)]) == 0
    assert publication.read_bytes() == built_receipt

    options = calls["build"]
    assert options.stage_threads == build_threads
    assert calls["run_threads"] == (8 if build_threads == 4 else 2)
    assert options.owners == options.owner_parts == parts
    assert options.parallel.owners == parts
    assert calls["run"]["runtime"] == options.runtime
    assert options.runtime.owner_count == parts
    assert options.runtime.cpu_device_count == parts
    launcher = {
        "launch_mode": launch,
        "rank_rss_limit_bytes": rss,
        "run_timeout_seconds": timeout,
    }
    assert {key: calls["run"][key] for key in launcher} == launcher
    scheduler = {key: value for key, value in calls["run"].items()
                 if key in ("rank_command", "rank_index_env", "collectives_interface")}
    assert scheduler == (dict(rank_command=("task-runner", "-n", "2"),
                              rank_index_env="TASK_INDEX", collectives_interface="cpu_net")
                         if launch == "scheduler" else {})


@pytest.mark.parametrize(
    "owner_setting, tables, error",
    [
        ("owners = 0", "", "owners must be a positive integer"),
        ("threads = 0", "", "threads must be a positive integer"),
        ("threads = -1", "", "threads must be a positive integer"),
        ("threads = true", "", "threads"),
        ("threads = 1.5", "", "threads"),
        ("owners = 5", "", "at least four eta planes per owner"),
        ("owners = 2", "[parallel]\nowners = 1", "owners"),
        ("owners = 2", "[parallel]\nexport_owner_q = true", "export_owner_q"),
        ("owners = 2", "[mms]\npartitions = 3", "partitions"),
        ("owners = 2", "[mms]\nhost_cpu_devices = 3", "host_cpu_devices"),
        ("", "[mms]\nface_record_chunk = 0", "face_record_chunk"),
        ("", "[mms]\nface_record_chunk = -1", "face_record_chunk"),
        ("", "[mms]\nface_record_chunk = true", "face_record_chunk"),
        ("", "[mms]\nface_record_chunk = 1.5", "face_record_chunk"),
        ("", '[mms]\ndevice = "tpu"', "device must be"),
        ("", "[mms]\ndevice = true", "device"),
        ("", '[mms]\ndevice = "gpu"\nlaunch_mode = "per_owner"', "GPU runs require local"),
        ("", '[mms]\ndevice = "gpu"\nlaunch_mode = "scheduler"', "GPU runs require local"),
        ("owners = 2", '[mms]\ndevice = "gpu"', "GPU runs require exactly one owner"),
        ("", "[mms]\nrank_rss_limit_bytes = 1024", "require launch_mode"),
        ("", "[mms]\nrun_timeout_seconds = 120", "require launch_mode"),
        ("owners = 2", '[mms]\nlaunch_mode = "scheduler"', "rank_command must"),
        ("owners = 2", '[mms]\ncollectives_interface = "cpu_net"', "require launch_mode"),
        ("owners = 2", '[mms]\nlaunch_mode = "scheduler"\nrank_command = "task-runner"',
         "rank_command must be a list"),
    ],
)
def test_owner_deck_rejects_conflicting_settings(tmp_path, owner_setting, tables, error):
    deck = write_deck(tmp_path, owner_setting, tables)
    with pytest.raises(SystemExit, match=error):
        main(["inspect", str(deck)])


@pytest.mark.parametrize("settings, expected", [
    ("", {}),
    ("face_record_chunk = 65536", {"face_record_chunk": 65536}),
    ('device = "cpu"', {}),
    ('device = "gpu"', {"device": "gpu"}),
])
def test_deck_forwards_only_configured_run_settings(tmp_path, monkeypatch, settings, expected):
    deck = write_deck(tmp_path, tables=f"[mms]\n{settings}")
    _, config = load_deck(deck)
    publication = tmp_path / "case/owner_inputs/receipt.json"
    publication.parent.mkdir(parents=True)
    publication.touch()
    monkeypatch.setattr(nodal_deck, "_check_publication", lambda *_: None)

    def run_mms(directory, options, *, stage_threads):
        assert {key: options[key] for key in ("face_record_chunk", "device") if key in options} == expected
        return {"steps": 1, "time": .001}

    monkeypatch.setattr(case, "run_mms", run_mms)
    nodal_deck.run(config, lambda _: None)


@pytest.mark.parametrize("key, value", [
    ("fit_sample_shape", "[8, 9, 8]"),
    ("metric_mesh_shape", "[64, 64, 64]"),
    ("eta_representation", '"spline"'),
    ("plane_construction", '"exact"'),
])
def test_owner_deck_rejects_retired_geometry_key(tmp_path, key, value):
    deck = write_deck(tmp_path)
    source = deck.read_text(encoding="utf-8")
    deck.write_text(source.replace("[field_filter]", f"{key} = {value}\n[field_filter]"),
                    encoding="utf-8")
    expected = rf"unknown \[geometry\] key\(s\): \['{key}'\]"
    with pytest.raises(SystemExit, match=expected):
        main(["inspect", str(deck)])
