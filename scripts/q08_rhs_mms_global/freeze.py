"""Local packaging only; never a remote failure recovery operation."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def main():
    files = sorted([p for p in HERE.rglob('*.py') if '__pycache__' not in p.parts]+[HERE/'README.md'])
    m = dict(schema='q08-six-field-static-mms-v1',
        baseline_identity='ff1bff86e4af5045b84856d16550f0f45edcbe03b83fc83cc5584557af4c8722',
        implementation_identity='1a953c60d7352b2a74e6a737d7c2be7dcca9ee194808db3154e5f840d4a9b087',
        n=[32,48,64], cases=22, kinds=4, diffusion_spans=[1/16,1/32],
        material_span=1/32, outer_span=1/16, actual_gpu_devices=4,
        reference_steps=[1e-4,5e-5,2.5e-5], method='polynomial',
        new_tracing=False,new_preparation=False,new_references=True,
        fields=['n','Te','Ti','Vi','Ve','omega'], outputs=31,
        replay_atol=1e-8,replay_rtol=1e-11,constant_atol=1e-7,
        ti_boundary_fixture_roundoff=dict(eps_multiplier=32,scale='max(1,abs(expected))',
            dtype='float64',nonwall_exact=True),
        files={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    p=HERE/'manifest.json';p.write_text(json.dumps(m,indent=2)+'\n')
    print(hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':main()
