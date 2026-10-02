"""Import the byte-frozen minimal Q runtime without production package imports."""
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
def configure(*, gpu=False, output=None):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['JAX_ENABLE_X64']='true'
    if gpu:
        os.environ.setdefault('JAX_PLATFORMS','cuda,cpu')
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    else:
        os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    if output:
        os.environ['JAX_COMPILATION_CACHE_DIR']=str(Path(output)/('jax_gpu_cache' if gpu else 'jax_cpu_cache'))
    sys.path[:0]=[str(HERE/'runtime'),str(REPO)]
