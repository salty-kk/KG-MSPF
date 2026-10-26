"""Run the fixed paper configuration in one-result subprocesses."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from filelock import FileLock

def jobs(stages, repeats=5, folds=5, draws=3):
    from experiment import MAIN_MODELS, ablation_variants, pam50_variants, prior_variants
    out=[]
    for stage in stages:
        if stage=='main':
            out += [(stage,m,f'r{r}_f{f}') for r in range(repeats) for f in range(1,folds+1) for m in MAIN_MODELS]
        elif stage in ('holdout','external'):
            out += [(stage,m,None) for m in MAIN_MODELS]
        elif stage=='postprocess':
            out.append((stage,None,None))
        else:
            variants={'ablation':ablation_variants,'pam50':pam50_variants,'prior':lambda:prior_variants(draws)}[stage]()
            variants={k:v for k,v in variants.items() if k!='molecular_offset'}
            out += [(stage,k,f'r0_f{f}') for f in range(1,folds+1) for k in variants]
    return out

def command(job,args,here,settings):
    stage,config,split=job
    if stage=='postprocess':return [sys.executable,str(here/'postprocess.py'),'--root',str(args.out)]
    cmd=[sys.executable,'-u',str(here/'run_experiments.py'),stage,'--out',str(args.out),'--device',args.device,
         '--folds',str(args.folds),'--draws',str(args.draws),'--strict','--save-checkpoints']
    for k,v in settings.items():cmd += ['--'+k.replace('_','-'),str(v)]
    if stage=='main':cmd += ['--repeats',str(args.repeats)]
    if stage=='external':cmd += ['--cohort','tcga','--data',str(args.external_data)]
    cmd += ['--models' if stage in ('main','holdout','external') else '--configs',config]
    if split:cmd += ['--split-names',split]
    return cmd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stages',default='main,holdout,ablation,pam50,prior')
    p.add_argument('--out',type=Path,default=Path('outputs/reproduction'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--draws',type=int,default=3)
    p.add_argument('--external-data',type=Path,default=Path('external_data/tcga_brca_harmonized.csv'))
    p.add_argument('--dry-run',action='store_true',help='List planned jobs without fitting models')
    args=p.parse_args();here=Path(__file__).resolve().parent
    stages=args.stages.split(',')
    if set(stages)-{'main','holdout','ablation','pam50','prior','external','postprocess'}:p.error('Unknown stage')
    if args.repeats<1 or args.folds<2 or args.draws<1:p.error('Invalid repeats, folds or draws')
    args.out=args.out.resolve();args.external_data=args.external_data.resolve()
    settings=json.loads((here/'paper_config.json').read_text(encoding='utf8'))['settings']
    plan=jobs(stages,args.repeats,args.folds,args.draws)
    if args.dry_run:
        print(json.dumps({'total':len(plan),'stages':{s:sum(j[0]==s for j in plan) for s in stages},'first_command':command(plan[0],args,here,settings)},indent=2));return
    if 'external' in stages and not args.external_data.exists():p.error('Prepare the TCGA CSV with fetch_tcga.py first')
    if not (here/'data/METABRIC_RNA_Mutation.csv').exists():p.error('Download METABRIC first; see data/README.md')
    args.out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for k,v in {'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','KGMSPF_BASELINE_CHUNK':'64','KGMSPF_RSF_JOBS':'1','PYTHONIOENCODING':'utf8'}.items():env.setdefault(k,v)
    with FileLock(str(args.out/'.run.lock'),timeout=0):
        logs=args.out/'logs';logs.mkdir(exist_ok=True)
        for i,job in enumerate(plan,1):
            print(f'{i}/{len(plan)} {job}',flush=True)
            label='_'.join(str(v) for v in job if v is not None)
            with (logs/(label+'.log')).open('a',encoding='utf8') as log:
                subprocess.run(command(job,args,here,settings),cwd=here,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Requested jobs finished. Run postprocess.py to generate aggregate tables and figures.')

if __name__=='__main__':main()
