"""Pedagogical model-selection guide for DKX users.

This example is intentionally lightweight. It teaches how to choose between
DKX model families, dimensions, fluid closures, and boundary-condition
families without launching a heavy validation campaign by default.

The script imports real public APIs, writes small starter TOML decks (plus a
Markdown guide and a machine-readable JSON summary under ``OUTPUT_ROOT``,
relative to the current working directory), parses those decks with the runtime
configuration layer, prints the full decision guide, and prints the API calls a
user would make when they are ready to inspect or run a case. Edit the
PARAMETERS constants below to change the output location or to actually run the
generated tiny diffusion deck.

Run from the repository root:

    PYTHONPATH=src python examples/model_selection_guide.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import dedent, indent
from typing import Any

from dkx.config.boutinp import load_bout_input
from dkx.native import run_input_case
from dkx.runtime import RunConfiguration


# --- PARAMETERS ------------------------------------------------------------------
OUTPUT_ROOT = Path("output/model_selection_guide")  # artifact root (cwd-relative)
WRITE_STARTER_DECKS = True         # write starter decks + JSON/Markdown summaries
RUN_TINY_DIFFUSION_SMOKE = False   # set True to actually run the tiny diffusion deck
QUIET = False                      # set True to write artifacts without printing the guide


@dataclass(frozen=True)
class ModelFamily:
    """One model family in the user-facing decision tree."""

    name: str
    use_when: str
    equations: tuple[str, ...]
    evolved_quantities: tuple[str, ...]
    dimensions: tuple[str, ...]
    boundary_choices: tuple[str, ...]
    starting_points: tuple[str, ...]
    caution: str


@dataclass(frozen=True)
class DeckSummary:
    """Metadata extracted by DKX from a generated starter deck."""

    path: str
    nout: int
    timestep: float
    mesh: dict[str, int | None]
    solver_type: str | None
    components: tuple[str, ...]


MODEL_FAMILIES = (
    ModelFamily(
        name="diffusion / scalar reduced transport",
        use_when=(
            "Start here for a fast executable deck, restart/output workflows, "
            "sensitivity studies, or a first cross-field transport prototype."
        ),
        equations=(
            "d_t n = div(D_perp grad n) + S_n",
            "d_t p = div(chi_perp grad p) + S_p, when pressure is evolved",
        ),
        evolved_quantities=("density", "pressure-like scalar"),
        dimensions=("1D radial", "2D slab", "thin 3D smoke tests"),
        boundary_choices=("Neumann/no-flux", "Dirichlet/fixed value", "periodic directions"),
        starting_points=(
            "examples/inputs/restartable_diffusion.toml",
            "examples/restartable_diffusion_tutorial.py",
        ),
        caution="This is not a sheath/recycling SOL model; it is a controlled reduced transport lane.",
    ),
    ModelFamily(
        name="full drift-reduced Braginskii-style open-field SOL",
        use_when=(
            "Use when parallel losses, sheath targets, recycling, density, "
            "pressure, and momentum closures determine the physics question."
        ),
        equations=(
            "d_t n_s + div(Gamma_s) = S_n,s",
            "d_t (n_s V_parallel,s) + div(Gamma_s V_parallel,s) = -grad_parallel p_s + F_s",
            "d_t p_s + div(p_s u_s) + gamma p_s div(u_s) = Q_s",
            "omega = div(C grad_perp phi), on electrostatic/vorticity lanes",
        ),
        evolved_quantities=("species density", "parallel momentum", "pressure", "phi/vorticity closures"),
        dimensions=("1D open-field line", "2D tokamak/open-field", "3D FCI closures where promoted"),
        boundary_choices=("Bohm/sheath targets", "target recycling", "neutral source/sink", "no-flow guards"),
        starting_points=(
            "examples/engineering/target_recycling_campaign_demo.py",
            "examples/engineering/neutral_mixed_boundary_campaign_demo.py",
            "docs/physics_models.md",
        ),
        caution=(
            "Use the documented curated/campaign examples for validated sheath, "
            "recycling, and neutral setup rather than inventing ad-hoc deck keys."
        ),
    ),
    ModelFamily(
        name="2D electrostatic drift-wave / blob / vorticity models",
        use_when=(
            "Use when the main question is reduced electrostatic turbulence, "
            "interchange drive, vorticity closure, or a compact benchmark."
        ),
        equations=(
            "d_t n + u_E dot grad n = parallel/current/source terms",
            "d_t omega + u_E dot grad omega = curvature + parallel-current + source terms",
            "u_E = b x grad phi / B",
        ),
        evolved_quantities=("density", "potential", "vorticity", "selected benchmark fields"),
        dimensions=("2D slab", "2D tokamak cross-section", "selected 3D field surfaces"),
        boundary_choices=("periodic", "zero-gradient", "zero-Dirichlet potential/vorticity guards"),
        starting_points=(
            "src/dkx/validation/blob2d.py",
            "src/dkx/validation/alfven_wave.py",
            "src/dkx/validation/drift_wave.py",
        ),
        caution="These are reduced benchmark lanes, not a substitute for full open-field recycling closure.",
    ),
    ModelFamily(
        name="3D geometry / FCI / selected-field workflows",
        use_when=(
            "Use when non-axisymmetric geometry, traced field lines, connection "
            "length, or selected 3D tokamak/stellarator fields are the object of study."
        ),
        equations=(
            "field-aligned derivatives and map exits supply parallel transport geometry",
            "sheath/recycling/neutral closures can be evaluated on open map endpoints",
        ),
        evolved_quantities=("geometry metrics", "selected fields", "FCI diagnostics", "compact reduced histories"),
        dimensions=("3D tokamak", "3D stellarator", "traced field-line coordinates"),
        boundary_choices=("open field-line endpoints", "periodic toroidal/field-line maps", "geometry-driven target masks"),
        starting_points=(
            "examples/geometry-3D/stellarator-fci/geometry_plotting.py",
            "examples/geometry-3D/stellarator-fci/nonlinear_turbulence.py",
            "examples/tokamak-3D/tcv-x21/selected_field_parity_demo.py",
        ),
        caution="Expect higher setup cost; start with release-backed examples before regenerating external geometry.",
    ),
)


DIMENSION_GUIDE = {
    "1D": "Use for closure development, parallel/open-field line scans, MMS, and cheap parameter sweeps.",
    "2D": "Use when perpendicular structure, blobs, drift waves, or diverted cross-sections matter.",
    "3D": "Use only when geometry is part of the question; otherwise 2D is cheaper and easier to validate.",
}

FLUID_GUIDE = {
    "one-fluid or scalar proxy": (
        "Choose for diffusion tutorials, bulk transport prototypes, and cases "
        "where separate electron/ion dynamics are not the question."
    ),
    "two-fluid / multispecies": (
        "Choose for sheath current balance, electron-ion heat exchange, "
        "separate ion/electron pressure, collisions, reactions, and recycling."
    ),
}

BOUNDARY_GUIDE = {
    "diffusion/no-flux": "Use Neumann-style field boundaries for isolated reduced transport tests.",
    "fixed-value": "Use Dirichlet-style boundaries when a benchmark specifies a fixed edge value.",
    "periodic": "Use for homogeneous slab directions or toroidal/poloidal benchmark directions.",
    "sheath": "Use at material targets when Bohm outflow and sheath energy losses are part of the model.",
    "recycling": "Use with sheath targets when target ion flux should return as neutral source.",
    "neutral mixed/diffusion": "Use when neutral density/energy transport and ionization/recombination are active.",
}


def diffusion_starter_deck() -> str:
    """Return a tiny runnable reduced-transport deck."""

    return (
        dedent(
            """
            [time]
            nout = 1
            timestep = 0.25

            [runtime]
            precision = "float64"

            [mesh]
            nx = 8
            ny = 6
            nz = 1
            dx = 0.05
            dy = 0.05
            dz = 1.0
            J = 1

            [solver]
            type = "native"
            mxstep = 100

            [model]
            components = ["h"]

            [species.h]
            type = ["evolve_density", "evolve_pressure", "anomalous_diffusion"]
            AA = 1
            charge = 1
            anomalous_D = 0.2
            thermal_conduction = false

            [fields.Nh]
            function = { expr = "1 + 0.1 * exp(-((x - 0.5)^2) / 0.02)" }
            bndry_all = "neumann"

            [fields.Ph]
            function = { ref = "Nh:function" }
            bndry_all = "neumann"
            """
        ).strip()
        + "\n"
    )


def open_field_concept_deck() -> str:
    """Return a parseable concept deck for higher-fidelity SOL choices.

    This deck is deliberately marked as a concept deck. It teaches component
    selection and boundary families, but users should start from the validated
    recycling/neutral examples before running a production open-field case.
    """

    return (
        dedent(
            """
            # Concept deck: parse this to inspect model metadata, then replace it
            # with a validated recycling/neutral campaign deck before production use.
            [time]
            nout = 1
            timestep = 0.05

            [runtime]
            precision = "float64"

            [mesh]
            nx = 8
            ny = 8
            nz = 1
            dx = 0.02
            dy = 0.05
            dz = 1.0
            J = 1

            [solver]
            type = "native"
            mxstep = 100

            [model]
            components = ["d", "e", "n"]

            [species.d]
            type = ["evolve_density", "evolve_pressure", "evolve_momentum", "anomalous_diffusion"]
            AA = 2
            charge = 1
            anomalous_D = 0.1

            [species.e]
            type = ["evolve_density", "evolve_pressure", "evolve_momentum"]
            AA = 5.446e-4
            charge = -1

            [species.n]
            type = ["evolve_density", "evolve_pressure", "neutral_diffusion"]
            AA = 2
            charge = 0

            [fields.Nd]
            function = { expr = "1.0" }
            bndry_all = "neumann"

            [fields.Ne]
            function = { ref = "Nd:function" }
            bndry_all = "neumann"

            [fields.Nn]
            function = { expr = "0.05" }
            bndry_all = "neumann"
            """
        ).strip()
        + "\n"
    )


def validate_deck(path: Path) -> DeckSummary:
    """Load a deck with DKX and extract runtime metadata."""

    config = load_bout_input(path)
    run_config = RunConfiguration.from_config(config)
    return DeckSummary(
        path=str(path),
        nout=run_config.time.nout,
        timestep=run_config.time.timestep,
        mesh={
            "nx": run_config.mesh.nx,
            "ny": run_config.mesh.ny,
            "nz": run_config.mesh.nz,
        },
        solver_type=run_config.solver.type,
        components=tuple(component.label for component in run_config.components),
    )


def write_starter_artifacts(output_root: Path) -> tuple[DeckSummary, ...]:
    """Write starter decks, a markdown guide, and a machine-readable summary."""

    output_root.mkdir(parents=True, exist_ok=True)
    deck_paths = {
        "diffusion_start.toml": diffusion_starter_deck(),
        "open_field_concept.toml": open_field_concept_deck(),
    }
    summaries = []
    for filename, text in deck_paths.items():
        path = output_root / filename
        path.write_text(text, encoding="utf-8")
        summaries.append(validate_deck(path))

    summary_payload = {
        "model_families": [asdict(family) for family in MODEL_FAMILIES],
        "dimension_guide": DIMENSION_GUIDE,
        "fluid_guide": FLUID_GUIDE,
        "boundary_guide": BOUNDARY_GUIDE,
        "generated_decks": [asdict(summary) for summary in summaries],
    }
    (output_root / "model_selection_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "model_selection_guide.md").write_text(
        render_markdown(summaries), encoding="utf-8"
    )
    return tuple(summaries)


def render_markdown(deck_summaries: tuple[DeckSummary, ...]) -> str:
    """Render the educational guide as portable Markdown."""

    lines = [
        "# DKX Model Selection Guide",
        "",
        "## Decision Tree",
        "",
        "- Start with diffusion/reduced transport for a runnable scalar prototype.",
        "- Move to 2D drift-wave/blob/vorticity models for electrostatic turbulence questions.",
        "- Move to drift-reduced Braginskii open-field lanes when sheath, recycling, momentum, and pressure closure matter.",
        "- Move to 3D FCI or selected-field workflows only when geometry is the question.",
        "",
        "## Model Families",
        "",
    ]
    for family in MODEL_FAMILIES:
        lines.extend(
            [
                f"### {family.name}",
                "",
                family.use_when,
                "",
                "Equations:",
                *[f"- `{equation}`" for equation in family.equations],
                "",
                f"Evolved quantities: {', '.join(family.evolved_quantities)}.",
                f"Useful dimensions: {', '.join(family.dimensions)}.",
                f"Boundary families: {', '.join(family.boundary_choices)}.",
                f"Caution: {family.caution}",
                "",
                "Starting points:",
                *[f"- `{path}`" for path in family.starting_points],
                "",
            ]
        )

    lines.extend(["## Generated Decks", ""])
    for summary in deck_summaries:
        lines.extend(
            [
                f"- `{summary.path}`",
                f"  Components: {', '.join(summary.components)}",
                f"  Mesh: {summary.mesh['nx']} x {summary.mesh['ny']} x {summary.mesh['nz']}",
            ]
        )
    return "\n".join(lines) + "\n"


def print_mapping(title: str, mapping: dict[str, str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in mapping.items():
        print(f"{key}: {value}")


def print_model_guide() -> None:
    print("\nDKX Model Selection")
    print("======================")
    for family in MODEL_FAMILIES:
        print(f"\n{family.name}")
        print("-" * len(family.name))
        print(family.use_when)
        print("Equations:")
        for equation in family.equations:
            print(f"  {equation}")
        print(f"Start from: {', '.join(family.starting_points)}")
        print(f"Caution: {family.caution}")

    print_mapping("Dimension choice", DIMENSION_GUIDE)
    print_mapping("Fluid closure choice", FLUID_GUIDE)
    print_mapping("Boundary choice", BOUNDARY_GUIDE)


def print_api_discovery(output_root: Path, deck_summaries: tuple[DeckSummary, ...]) -> None:
    examples = {
        "inspect a deck": f"dkx inspect {output_root / 'diffusion_start.toml'}",
        "run a deck": f"dkx run {output_root / 'diffusion_start.toml'} --verbose",
        "load deck in Python": "config = load_bout_input(path)",
        "inspect runtime metadata": "run_config = RunConfiguration.from_config(config)",
        "run from Python": "result = run_input_case(path, case_name='my_case', parity_mode='run')",
        "discover public native API": "python -c \"import dkx.native as n; print(n.__all__)\"",
    }
    print_mapping("API discovery and use", examples)

    if deck_summaries:
        print("\nParsed starter deck metadata")
        print("----------------------------")
        for summary in deck_summaries:
            print(f"{summary.path}")
            print(indent(json.dumps(asdict(summary), indent=2), "  "))


def maybe_run_tiny_diffusion(deck_path: Path) -> dict[str, Any]:
    """Optionally exercise the generated reduced-transport deck."""

    result = run_input_case(deck_path, case_name="model_selection_diffusion", parity_mode="run", output_steps=1)
    return {
        "time_points": [float(value) for value in result.time_points],
        "variables": sorted(result.variables),
    }


# --- write starter decks, print the guide, optionally run the tiny deck -----------
DECK_SUMMARIES: tuple[DeckSummary, ...] = ()
if WRITE_STARTER_DECKS:
    DECK_SUMMARIES = write_starter_artifacts(OUTPUT_ROOT)
    print(f"wrote starter decks and summaries under {OUTPUT_ROOT}")
if not QUIET:
    print_model_guide()
    print_api_discovery(OUTPUT_ROOT, DECK_SUMMARIES)
if RUN_TINY_DIFFUSION_SMOKE:
    RESULT = maybe_run_tiny_diffusion(OUTPUT_ROOT / "diffusion_start.toml")
    print_mapping(
        "Tiny diffusion smoke result",
        {key: str(value) for key, value in RESULT.items()},
    )
