#!/usr/bin/env python3
"""Windows-friendly IN-CXR full pipeline using the original uploaded scripts."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLIT_FILES = ('candidates.csv', 'labels.npy', 'keep.npy', 'target_keep.npy')

def execute(cmd, log):
    print('\n' + '='*76 + '\nRUN: ' + ' '.join(map(str,cmd)) + '\n' + '='*76,flush=True)
    with log.open('a',encoding='utf-8') as f:
        f.write('\nRUN: '+ ' '.join(map(str,cmd))+'\n')
        p = subprocess.Popen([str(v) for v in cmd], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line,end='',flush=True); f.write(line)
        ret = p.wait()
    if ret: raise SystemExit(f'Stage failed ({ret}). See {log}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path(r'D:\RMIA_IN_CXR'))
    ap.add_argument('--data-root',type=Path,default=None,
                    help='Folder directly containing Normal and AbNormal folders (default PROJECT/data/IN-CXR)')
    ap.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--infer-batch-size',type=int,default=32)
    ap.add_argument('--x-chunk',type=int,default=64)
    ap.add_argument('--target-batch-size',type=int,default=64)
    ap.add_argument('--no-amp',action='store_true')
    ap.add_argument('--regenerate-splits',action='store_true',help='ONLY for a completely new experiment; unsafe with existing checkpoints')
    a=ap.parse_args()
    project=a.project.resolve(); root=(a.data_root or project/'data'/'IN-CXR').resolve()
    if not (root/'Normal').is_dir() or not (root/'AbNormal').is_dir():
        raise SystemExit(f'Expected Normal and AbNormal directories directly inside: {root}')
    exp=project/'exp'/'incxr_16'; exp.mkdir(parents=True,exist_ok=True)
    results=project/'results'/'incxr_densenet121'; target=results/'target'/'target_densenet121.pt'
    log=project/'run_incxr_full.log'
    for name in ('prepare_incxr.py','incxr_densenet121_lossmia.py','incxr_densenet121_rmia_fixed.py'):
        if not (HERE/name).is_file(): raise SystemExit(f'Missing script next to runner: {name}')
    present=[(exp/n).exists() for n in SPLIT_FILES]
    if a.regenerate_splits:
        if target.exists() or (results/'rmia'/'models').exists() or (results/'rmia'/'inference').exists():
            raise SystemExit('Refusing to regenerate membership masks with models or logits present. Use a NEW project directory.')
        present=[False]*4
    if any(present) and not all(present):
        raise SystemExit('Partial split artifacts found; refusing to mix or overwrite. Move them aside or restore all 4 files.')
    if not all(present):
        print('Generating reproducible candidate order and membership masks...',flush=True)
        spec=importlib.util.spec_from_file_location('prepare_incxr_module',HERE/'prepare_incxr.py')
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.ROOT=root; mod.OUT=exp; mod.main()
    else:
        print('Reusing existing candidate order and membership masks. No regeneration.',flush=True)
    import numpy as np
    labels=np.load(exp/'labels.npy'); keep=np.load(exp/'keep.npy'); tkeep=np.load(exp/'target_keep.npy')
    if len(labels)!=12600 or keep.shape!=(16,len(labels)) or tkeep.shape!=(len(labels),) or not np.all(keep.sum(axis=0)==8):
        raise SystemExit(f'Unexpected cohort/membership shapes: labels={labels.shape}, keep={keep.shape}, target={tkeep.shape}')
    if int(tkeep.sum())!=6300: raise SystemExit(f'Expected 6300 target members; found {int(tkeep.sum())}')
    # Both imported scripts use PROJECT/data/IN-CXR for image resolution. If external root is provided,
    # create a directory junction so source code and original relative paths remain unchanged.
    expected_root=(project/'data'/'IN-CXR').resolve()
    if root != expected_root:
        if expected_root.exists():
            if expected_root.resolve()!=root: raise SystemExit(f'Existing data path differs: {expected_root}; set --data-root appropriately.')
        else:
            expected_root.parent.mkdir(parents=True,exist_ok=True)
            import os
            if sys.platform!='win32': raise SystemExit('External data-root junction requires Windows; place dataset inside PROJECT/data/IN-CXR.')
            execute(['cmd','/c','mklink','/J',str(expected_root),str(root)],log)
    common=[sys.executable,'-u']
    if not target.exists():
        cmd=common+[HERE/'incxr_densenet121_lossmia.py','--data-root',expected_root,
                    '--split-dir',exp,'--output-dir',results,'--epochs','100',
                    '--batch-size',str(a.target_batch_size),'--num-workers',str(a.workers)]
        if a.no_amp: cmd+=['--no-amp']
        execute(cmd,log)
    else:
        print(f'Reusing existing target checkpoint: {target}',flush=True)
    cmd=common+[HERE/'incxr_densenet121_rmia_fixed.py','--project',project,
                '--stage','all','--device',a.device,'--attack-device',a.device,
                '--workers',str(a.workers),'--infer-batch-size',str(a.infer_batch_size),
                '--x-chunk',str(a.x_chunk)]
    if a.no_amp: cmd+=['--no-amp']
    execute(cmd,log)
    metric=results/'rmia'/'attack'/'rmia_metrics.json'
    print(f'\nCOMPLETE. RMIA metrics: {metric}\nLog: {log}',flush=True)

if __name__=='__main__': main()
