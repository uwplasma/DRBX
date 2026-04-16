from __future__ import annotations

import json
from pathlib import Path


def build_hermes_capability_audit() -> dict[str, object]:
    families = [
        {
            "family": "core_scheduler_and_normalization",
            "reference_surfaces": ["component_scheduler", "normalization", "metric loading"],
            "jax_status": "closed",
            "jax_capability": "native_exact",
            "evidence": [
                "scheduler transform/finally contract mirrored",
                "normalization and metric scaling locked in committed baselines",
            ],
            "next_gate": "keep covered through manifest/reference payload regression",
        },
        {
            "family": "fluid_1d_mms",
            "reference_surfaces": ["1D-fluid integrated test", "MMS convergence"],
            "jax_status": "closed",
            "jax_capability": "native_exact",
            "evidence": [
                "RHS, one-step, and short-window baselines committed",
                "manufactured-solution convergence campaign committed",
            ],
            "next_gate": "use in reviewer-facing convergence package",
        },
        {
            "family": "diffusion_transport",
            "reference_surfaces": ["diffusion integrated test"],
            "jax_status": "closed",
            "jax_capability": "native_exact",
            "evidence": [
                "one-step and short-window exact ladders committed",
            ],
            "next_gate": "keep as low-level transport regression substrate",
        },
        {
            "family": "vorticity_and_relax_potential",
            "reference_surfaces": ["vorticity integrated test"],
            "jax_status": "partially_closed",
            "jax_capability": "native_exact",
            "evidence": [
                "vorticity rhs/one-step/short-window exact ladders committed",
            ],
            "next_gate": "finish broader relax-potential boundary/dissipation options beyond benchmark slice",
        },
        {
            "family": "drift_wave_and_blob_benchmarks",
            "reference_surfaces": ["drift-wave integrated test", "blob benchmark package"],
            "jax_status": "partially_closed",
            "jax_capability": "native_validated",
            "evidence": [
                "drift-wave rhs/one-step and short-window benchmark package committed",
                "blob benchmark package committed",
            ],
            "next_gate": "promote long-window benchmark evidence into broader turbulence campaign bundle",
        },
        {
            "family": "neutral_mixed",
            "reference_surfaces": ["neutral_mixed integrated test", "neutral mixed benchmark diagnostics"],
            "jax_status": "open",
            "jax_capability": "native_rhs_exact_transient_open",
            "evidence": [
                "RHS parity is committed",
                "native transient runner path now exists for one-step and short-window",
            ],
            "next_gate": "close one-step and short-window parity against committed baselines before promotion",
        },
        {
            "family": "open_field_recycling",
            "reference_surfaces": ["1D-recycling", "1D-recycling-dthe"],
            "jax_status": "partially_closed",
            "jax_capability": "native_exact_one_step_native_operational_window",
            "evidence": [
                "single-species and multispecies rhs and one-step ladders promoted",
                "short-window ladder committed as operational evidence",
            ],
            "next_gate": "promote richer windows and long-run recycling closure",
        },
        {
            "family": "integrated_2d_recycling",
            "reference_surfaces": ["2D-recycling integrated test"],
            "jax_status": "closed",
            "jax_capability": "native_exact",
            "evidence": [
                "rhs, one-step, short-window, and medium-window ladders promoted",
            ],
            "next_gate": "keep exact gate stable while broader 2D campaigns land",
        },
        {
            "family": "direct_tokamak_recycling",
            "reference_surfaces": ["direct tokamak recycling rungs"],
            "jax_status": "partially_closed",
            "jax_capability": "native_exact_first_output",
            "evidence": [
                "direct D/T one-step compare surface promoted",
            ],
            "next_gate": "finish richer transient windows and distributed-guard evolution",
        },
        {
            "family": "tokamak_2d_transport_and_turbulence",
            "reference_surfaces": ["tokamak diffusion/transport/isothermal/turbulence ladders"],
            "jax_status": "closed",
            "jax_capability": "native_exact_selected_matrix",
            "evidence": [
                "isothermal and turbulence one-step/short-window exact ladders committed",
            ],
            "next_gate": "use as main 2D exact matrix in paper-grade campaign",
        },
        {
            "family": "electromagnetic_selected_benchmarks",
            "reference_surfaces": ["alfven-wave", "annulus-isothermal-he-emag"],
            "jax_status": "closed_selected",
            "jax_capability": "native_exact_selected_matrix",
            "evidence": [
                "Alfven and annulus EM ladders committed through short/medium windows",
            ],
            "next_gate": "keep claims restricted to selected EM benchmark lanes",
        },
        {
            "family": "tokamak_3d_reduced_native_rungs",
            "reference_surfaces": ["tokamak native selected-field one-step", "tokamak isothermal short-window selected-field"],
            "jax_status": "partially_closed",
            "jax_capability": "native_exact_reduced",
            "evidence": [
                "native selected-field bundles committed with runtime and comparison reports",
            ],
            "next_gate": "expand native 3D convergence, scaling, and runtime campaign",
        },
        {
            "family": "non_tokamak_3d_geometry_adapters",
            "reference_surfaces": ["traced-field-line selected-field", "stellarator VMEC selected-field"],
            "jax_status": "closed_selected",
            "jax_capability": "native_exact_reduced_selected_matrix",
            "evidence": [
                "explicit external-pair parity gates committed for traced-field-line and VMEC families",
                "native reduced traced-field-line and VMEC rungs committed on the shared 3D artifact surface",
            ],
            "next_gate": "broaden the native 3D convergence, scaling, and runtime campaign on the promoted reduced matrix",
        },
        {
            "family": "reactions_collisions_and_atomic_data",
            "reference_surfaces": ["ADAS/AMJUEL reactions", "collfreq-braginskii-afn", "collfreq-multispecies"],
            "jax_status": "partially_closed",
            "jax_capability": "dedicated_campaign_plus_selected_recycling_lanes",
            "evidence": [
                "hydrogen reaction/recycling semantics exercised on committed recycling ladders",
                "dedicated reactions/collisions verification campaign committed",
            ],
            "next_gate": "expand the dedicated campaign into impurity/radiation breadth and longer-window transient checks",
        },
        {
            "family": "impurity_radiation_and_detachment_control",
            "reference_surfaces": ["fixed_fraction_radiation", "temperature_feedback", "detachment_controller", "ADAS carbon/neon"],
            "jax_status": "partially_closed",
            "jax_capability": "dedicated_campaign_plus_neon_rhs_lane_plus_feedback_controller_gate",
            "evidence": [
                "dedicated impurity/radiation validation campaign committed for neon OpenADAS and D/T/He/Ne RHS closure",
                "dedicated controller-feedback campaign committed for the native upstream-density feedback history on recycling_1d_one_step",
                "reduced temperature-feedback campaign package is in-tree, but the bounded local Hermes Tt-control example still exceeds the five-minute validation policy",
                "controller-oriented temperature/detachment surfaces are still not promoted",
            ],
            "next_gate": "add the first promoted temperature-feedback or detachment-control native lane beyond the upstream-density feedback controller gate",
        },
        {
            "family": "sod_shock_and_2d_energy_regression_surfaces",
            "reference_surfaces": ["sod-shock", "sod-shock-energy", "2D-energy"],
            "jax_status": "open",
            "jax_capability": "missing",
            "evidence": [
                "no dedicated curated ladders committed for these integrated tests",
            ],
            "next_gate": "decide whether to implement or explicitly scope them out of first strong-subset claim",
        },
    ]
    closed = sum(1 for item in families if item["jax_status"] == "closed")
    partial = sum(1 for item in families if item["jax_status"] in {"partially_closed", "closed_selected"})
    open_count = sum(1 for item in families if item["jax_status"] == "open")
    return {
        "reference_code": "hermes-3",
        "family_count": len(families),
        "closed_family_count": closed,
        "partially_closed_family_count": partial,
        "open_family_count": open_count,
        "families": families,
        "remaining_priority_families": [
            "neutral_mixed",
            "open_field_recycling",
            "direct_tokamak_recycling",
            "non_tokamak_3d_geometry_adapters",
            "reactions_collisions_and_atomic_data",
            "impurity_radiation_and_detachment_control",
        ],
    }


def write_hermes_capability_audit(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_hermes_capability_audit(), indent=2, sort_keys=True), encoding="utf-8")
    return target
