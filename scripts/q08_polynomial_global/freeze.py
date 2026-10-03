"""Local maintainer packaging, never run as a remote recovery operation."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    files = [p for p in HERE.rglob('*.py') if '__pycache__' not in p.parts]
    files += [HERE / 'README.md']
    manifest = dict(schema='q08-polynomial-global-v1',
        baseline_identity='ff1bff86e4af5045b84856d16550f0f45edcbe03b83fc83cc5584557af4c8722',
        profile_identity='10ce76b4096cccfb7dd5dc6aacafbee15853685fe0ed2c5b4b421c7a96f6f849',
        method='polynomial', default_unchanged=True, n=[32,48,64], cases=22,
        kinds=4, diffusion_spans=[1/16,1/32], material_span=1/32,
        material_outer_span=1/16, devices=[1,4], repeats=7,
        atol=1e-8, rtol=1e-11, new_tracing=False, new_preparation=False,
        files={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    target = HERE / 'manifest.json'
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    print(hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
