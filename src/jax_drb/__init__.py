from .runtime import configure_jax_runtime

# Configure the default precision/cache policy as early as possible so user
# imports that touch JAX arrays through jax_drb keep the requested dtype.
configure_jax_runtime()

from .config.boutinp import BoutConfig, NumericResolver, ROOT_SECTION, load_bout_input, parse_bout_input
from .config.normalization import MetricPolicy, ModelNormalization

__all__ = [
    "BoutConfig",
    "MetricPolicy",
    "ModelNormalization",
    "NumericResolver",
    "ROOT_SECTION",
    "load_bout_input",
    "parse_bout_input",
]

__version__ = "1.0.1"
