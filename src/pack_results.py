from datetime import datetime,timezone
from pathlib import Path
import zipfile
from src.common import ROOT, read_json, write_json, sha256


def main():
    audit_path = ROOT/'summary'/'audit.json'
    if not audit_path.exists():
        raise RuntimeError('Run collect before packing')
    audit = read_json(audit_path)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = ROOT/'return_bundle'
    folder.mkdir(exist_ok=True)
    path = folder/f'GAT_Medical_RESULTS_{stamp}.zip'
    selected = []
    for sub in ('summary','configs','docs','src','logs','results/formal','cache/splits'):
        for file in sorted((ROOT/sub).rglob('*')):
            if not file.is_file() or '__pycache__' in file.parts:
                continue
            if file.name.startswith('.') or file.suffix in ('.pt','.npy','.joblib','.tmp'):
                continue
            selected.append(file)
    for relative in ('README_先读我.md','data/source_manifest.json','cache/audit.json',
                     'cache/cell_ids.json','cache/cold_cell_ids.json'):
        if (ROOT/relative).exists():
            selected.append(ROOT/relative)
    manifest = {file.relative_to(ROOT).as_posix():sha256(file) for file in selected}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in selected:
            archive.write(file,file.relative_to(ROOT).as_posix())
        import json
        archive.writestr('RETURN_MANIFEST.json',json.dumps({'files_sha256':manifest,
                           'publication_eligible':audit['publication_eligible'],
                           'checkpoints_remain_on_server':True},ensure_ascii=False,indent=2))
    write_json(folder/'latest_return.json',{'file':path.name,'sha256':sha256(path),
               'publication_eligible':audit['publication_eligible'],
               'complete_tasks':audit['audited_complete_tasks']})
    print(f'RETURN THIS FILE: {path}',flush=True)


if __name__=='__main__':
    main()
