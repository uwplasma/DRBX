"""Local packaging only; remote runs never regenerate this manifest."""
import hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NAMES=('q_plan','q_sharding','q_parallel_material','q_parallel_characteristic','q_characteristic_polynomial')


def main():
    (HERE/'overlay').mkdir(exist_ok=True)
    for name in NAMES:shutil.copy2(REPO/'src/drbx/native'/f'{name}.py',HERE/'overlay'/f'{name}.py')
    files=[HERE/'profile.py',*(HERE/'overlay').glob('*.py')]
    receipt=dict(schema='q08-polynomial-profile-v1',default_unchanged=True,backend='gpu',
        cases=[1,0,21],kinds=[0,1,2,3],n=32,diffusion_span=1/16,material_span=1/32,
        atol=1e-8,rtol=1e-11,repeats=3,devices=[1,4],
        files={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    (HERE/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest())


if __name__=='__main__':main()
