"""Freeze geometry implementation and immutable HSX input identities."""
import argparse, gzip, hashlib, io, json, tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input-root',type=Path,required=True);args=p.parse_args()
    config=json.loads((HERE/'configuration.json').read_text())
    paths=[config['metric_cache'],config['makegrid']]
    for n in config['resolutions']:
        paths.extend(f"{config['geometry']}/{n}x{n}x{n}/{file}" for file in ('base_geometry.npz','rlp_topology.npz'))
    manifest={'files':[{'path':r,'bytes':(args.input_root/r).stat().st_size,'sha256':sha(args.input_root/r)} for r in paths]}
    (HERE/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files={};raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w') as tar:
        for path in sorted((REPO/'src/drbx').rglob('*')):
            if not path.is_file() or path.suffix not in ('.py','.json','.toml') or any(x in path.parts for x in ('__pycache__','dev_docs')):continue
            rel=str(path.relative_to(REPO));payload=path.read_bytes();files[rel]=hashlib.sha256(payload).hexdigest()
            info=tarfile.TarInfo(rel);info.size=len(payload);info.mode=0o644;tar.addfile(info,io.BytesIO(payload))
    archive=HERE/'geometry_source.tar.gz';archive.write_bytes(gzip.compress(raw.getvalue(),mtime=0))
    (HERE/'geometry_source_manifest.json').write_text(json.dumps({'archive_sha256':sha(archive),'files':files},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'source_files':len(files),'archive_bytes':archive.stat().st_size,'input_bytes':sum(x['bytes'] for x in manifest['files'])}))
if __name__=='__main__':main()
