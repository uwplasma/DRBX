"""Campaign-only minimal runtime import; no production package initialization."""
from pathlib import Path
import os
import sys
import types

HERE = Path(__file__).resolve().parent


def configure(run, *, gpu):
    if 'jax' in sys.modules or 'drbx' in sys.modules:
        raise RuntimeError('configure Q09 runtime before importing JAX/drbx')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    os.environ['JAX_ENABLE_X64'] = 'true'
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu' if gpu else 'cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(Path(run)/('cache_gpu' if gpu else 'cache_cpu'))
    os.environ['DRBX_CACHE_DIR'] = str(Path(run)/'cache_drbx')
    # Preserve scheduler CUDA visibility. GPU stages must prove actual hardware.
    root = HERE/'runtime/drbx'
    if not root.is_dir():
        raise ValueError('frozen Q09 runtime missing')
    for name, path in (('drbx', root), ('drbx.native', root/'native'),
                       ('drbx.stencils', root/'stencils'), ('drbx.geometry', root/'geometry')):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]; package.__file__ = str(path/'__init__.py')
        sys.modules[name] = package
