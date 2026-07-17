from __future__ import annotations

from .boutinp import BoutConfig

PUBLIC_MODEL_SECTION = "model"


def locate_model_section(config: BoutConfig) -> str:
    if config.has_section(PUBLIC_MODEL_SECTION):
        return PUBLIC_MODEL_SECTION

    component_sections = tuple(
        name for name in config.section_names() if config.has_option(name, "components")
    )
    if len(component_sections) == 1:
        return component_sections[0]

    normalized_sections = tuple(
        name
        for name in component_sections
        if all(config.has_option(name, key) for key in ("Nnorm", "Tnorm", "Bnorm"))
    )
    if len(normalized_sections) == 1:
        return normalized_sections[0]

    raise KeyError("Could not determine the model section. Define [model] or provide a single component section.")


def has_model_section(config: BoutConfig) -> bool:
    try:
        locate_model_section(config)
    except KeyError:
        return False
    return True
