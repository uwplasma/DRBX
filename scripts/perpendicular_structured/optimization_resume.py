"""Explicit, release-pinned execution upgrade without rewriting old checkpoints.

The original manifest is the numerical lineage. A separate certificate pins the
new execution sources. Both identities must verify before either can be reused.
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POLICY = Path(__file__).with_name('optimization_release.json')
CERTIFICATE = 'optimization_execution.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def check(output, kind, manifest, sources, commit=None):
    path=Path(output)/CERTIFICATE
    if not path.exists():
        raise ValueError('source identity changed; explicit adopt-optimization required')
    certificate=json.loads(path.read_text())
    if (certificate['kind']!=kind or certificate['manifest_hash']!=canonical(manifest)
        or certificate['execution_sources']!=sources
        or certificate['policy_sha256']!=sha(POLICY)
        or (commit is not None and certificate['execution_commit']!=commit)):
        raise ValueError('optimization execution certificate mismatch')
    return certificate


def adopt(output, kind, manifest, sources, commit):
    policy=json.loads(POLICY.read_text())
    rule=policy['campaigns'][kind]
    baseline=manifest['content'] if kind=='p05' else manifest['content_identity']
    if sources!=rule['execution_sources'] or baseline['sources']!=rule['baseline_sources']:
        raise ValueError('source revision is not in the tested optimization release')
    config_key='config' if kind=='p05' else 'configuration'
    if (baseline[config_key]!=rule['configuration'] or baseline['inputs']!=rule['inputs']
        or (kind=='p05' and baseline['commit']!=policy['baseline_commit'])):
        raise ValueError('baseline numerical configuration/input identity changed')
    expected=canonical(baseline)
    if expected != (manifest['identity'] if kind=='p05' else manifest['sha256']):
        raise ValueError('invalid baseline manifest digest')
    certificate=dict(schema='drbx.structured-optimization-execution-v1',kind=kind,
                     manifest_hash=canonical(manifest),policy_sha256=sha(POLICY),
                     baseline_commit=policy['baseline_commit'],execution_commit=commit,
                     execution_sources=sources,
                     note='Original manifests/checkpoints retained; equivalent optimized execution explicitly authorized.')
    path=Path(output)/CERTIFICATE
    if path.exists():
        return check(output,kind,manifest,sources,commit if kind=='p05' else None)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(certificate,indent=2,sort_keys=True)+'\n')
    temporary.replace(path)
    return certificate


def legacy_kernel_identity(output, current):
    """P06's nested identities include file sizes and absolute paths as well."""
    path=Path(output)/CERTIFICATE
    if not path.exists():
        return current
    certificate=json.loads(path.read_text())
    policy=json.loads(POLICY.read_text())
    if certificate['kind']!='p06' or certificate['policy_sha256']!=sha(POLICY):
        raise ValueError('invalid P06 optimization execution certificate')
    rule=policy['campaigns']['p06']
    result={}
    for name,record in current.items():
        file=Path(record['path'])
        if name!='reference_sidecar' and file.is_relative_to(REPO):
            relative=str(file.relative_to(REPO))
            if record['sha256']!=rule['execution_sources'].get(relative):
                raise ValueError('P06 execution source changed: '+relative)
            if relative not in rule['baseline_sources']:
                continue
            record={**record,'sha256':rule['baseline_sources'][relative],
                    'bytes':policy['baseline_bytes'][relative]}
        result[name]=record
    return result


def execution_provenance(output):
    path=Path(output)/CERTIFICATE
    if not path.exists():
        return None
    certificate=json.loads(path.read_text())
    return dict(certificate_sha256=sha(path),commit=certificate['execution_commit'],
                policy_sha256=certificate['policy_sha256'])
