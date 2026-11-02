"""Resumable unified benchmark. Run: python exp_unified.py [--dataset all].

Legacy results/checkpoints are never reused. All output stays in
results/unified; inputs are read without modifying old caches/manuscript.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import io
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import importlib.metadata
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.stats import rankdata
import openpyxl
import torch
from torch.utils.data import DataLoader, Subset
import protocol as P
from data import BatteryWindowDataset, WindowConfig, compute_norm_stats
from data_calce import CALCE_FILES, CALCE_RAW_DIR
from ecm import ECMParams
from ecm_2rc import ECMParams2RC
from models import LSTMBaseline
from training import TrainConfig, train, predict
from utils import set_seed
from unified_dynamics import UnifiedECMUDE

ROOT=Path(__file__).resolve().parent

def now():return datetime.now(timezone.utc).isoformat()

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for f in self.streams:f.write(s);f.flush()
        return len(s)
    def flush(self):
        for f in self.streams:f.flush()

@contextlib.contextmanager
def log_file(name):
    path=P.OUTPUT/'logs'/f'{name}.log';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',buffering=1) as f:
        with contextlib.redirect_stdout(Tee(sys.stdout,f)), contextlib.redirect_stderr(Tee(sys.stderr,f)):
            print('START',name,now(),flush=True)
            try:yield
            except BaseException:
                traceback.print_exc();print('FAILED',name,now());raise
            else:print('COMPLETE',name,now(),flush=True)

def ocv_calibration():
    """Independent 25C low-current discharge, never a dynamic test target."""
    path=CALCE_RAW_DIR/'SP1_25C_LC_OCV_11_5_2015'/'11_5_2015_low current OCV test_SP20-1.xlsx'
    if not path.exists():return None,dict(available=False,reason='independent calibration missing')
    wb=openpyxl.load_workbook(path,read_only=True,data_only=True)
    rows=[r for r in wb.active.values if len(r)>=4 and isinstance(r[0],(float,int)) and r[1]==3]
    wb.close();a=np.asarray([r[:4] for r in rows],dtype=float)
    t=a[:,0];i=-a[:,3]/1000.;v=a[:,2]/1000.
    ah=np.r_[0.,np.cumsum((i[1:]+i[:-1])*.5*np.diff(t)/3600.)]
    z=P.causal_soc(ah,qnom_Ah=2.,z0=1.)
    # Bin to suppress millivolt quantization and enforce monotonic OCV.
    zg=[];vg=[]
    for lo in np.linspace(0,1,501)[:-1]:
        m=(z>=lo)&(z<lo+.002)
        if m.any():zg.append(float(np.median(z[m])));vg.append(float(np.median(v[m])))
    zg=np.asarray(zg);vg=np.maximum.accumulate(vg)
    unique=np.r_[True,np.diff(vg)>0]
    return (vg[unique],zg[unique]),dict(available=True,path=str(path),sha256=P.sha256(path),
        method='SP1 independent 25C 0.05C discharge; trapezoidal current integration, Qnom=2Ah, z0=1; 0.002 SOC median bins; monotonic cumulative maximum',
        limitation='low-current OCV approximation, different cell; fixed 25C curve reused zero-shot at all temperatures',
        voltage_range_V=[float(vg.min()),float(vg.max())])

def initial_soc(cfg,voltage,rest_duration,calibration):
    if calibration is not None and voltage is not None and rest_duration>=P.REST_SECONDS:
        v,z=calibration
        if v[0]<=voltage<=v[-1]:
            return float(np.interp(voltage,v,z)),dict(method='inverse_independent_OCV',rest_voltage_V=voltage,rest_duration_s=rest_duration)
    return cfg.default_z0,dict(method='documented_fallback',reason='no independent OCV calibration and qualifying pre-cycle rest, or voltage outside calibration range',
                               rest_voltage_V=voltage,rest_duration_s=rest_duration)

def load_panasonic(name,cfg):
    path=Path('datasets')/P.PANASONIC_FILES[name]
    m=loadmat(path,squeeze_me=True,struct_as_record=False)['meas']
    t=np.asarray(m.Time,dtype=float).ravel()
    arrays=[np.asarray(getattr(m,k),dtype=float).ravel()
            for k in ('Current','Voltage','Battery_Temp_degC','Ah')]
    order=np.argsort(t,kind='stable');t=t[order];arrays=[a[order] for a in arrays]
    t,ix=np.unique(t,return_index=True);arrays=[a[ix] for a in arrays]
    # Remove only the initial thermal-soak block, retaining driving data.
    gaps=np.flatnonzero(np.diff(t)>5.)
    trim=0
    if len(gaps) and gaps[0]==0 and np.all(np.abs(arrays[0][:gaps[-1]+1])<.05):
        trim=int(gaps[-1]+1);t=t[trim:];arrays=[a[trim:] for a in arrays]
    t=t-t[0];grid=np.arange(int(np.floor(t[-1]/cfg.dt_seconds))+1)*cfg.dt_seconds
    i,v,temp,ah=[np.interp(grid,t,a) for a in arrays]
    z0,meta=initial_soc(cfg,None,0.,None)
    # Panasonic exports charge-positive net Ah; canonical Ah is discharge-positive.
    net=-ah
    return make_cycle(grid,i,v,temp,net,z0,cfg),dict(path=str(path),sha256=P.sha256(path),
        raw_Ah_sign='charge-positive, negated before protocol SOC',trimmed_samples=trim,
        sampling='sort, deduplicate timestamps, linear resample uniform grid',z0=z0,z0_estimation=meta)

def load_calce(name,cfg,calibration):
    folder,filename,temp=CALCE_FILES[name];path=CALCE_RAW_DIR/folder/filename
    wb=openpyxl.load_workbook(io.BytesIO(path.read_bytes()),read_only=True,data_only=True)
    sheet=wb[wb.sheetnames[1] if len(wb.sheetnames)>1 else wb.sheetnames[0]]
    dynamic=[];rest=[]
    for r in sheet.values:
        if len(r)<10 or not isinstance(r[4],(int,float)):continue
        step=int(r[4])
        if step==7:dynamic.append((r[1],r[6],r[7],r[9]-r[8]))
        elif step==6 and abs(float(r[6]))<=P.REST_CURRENT_A:rest.append((float(r[1]),float(r[7])))
    wb.close();a=np.asarray(dynamic,dtype=float);a=a[np.argsort(a[:,0],kind='stable')]
    _,ix=np.unique(a[:,0],return_index=True);a=a[ix]
    t0=a[0,0];rest=[r for r in rest if r[0]<=t0]
    vr=rest[-1][1] if rest else None
    duration=rest[-1][0]-rest[0][0] if len(rest)>1 else 0.
    z0,meta=initial_soc(cfg,vr,duration,calibration)
    t=a[:,0]-t0;grid=np.arange(int(np.floor(t[-1]/cfg.dt_seconds))+1)*cfg.dt_seconds
    i,v,net=[np.interp(grid,t,a[:,j]) for j in (1,2,3)]
    return make_cycle(grid,i,v,np.full(len(grid),temp),net,z0,cfg),dict(path=str(path),sha256=P.sha256(path),
        raw_Ah_sign='Discharge_Capacity minus Charge_Capacity (includes regeneration)',
        sampling='Step 7 sorted/deduplicated, uniform linear interpolation',z0=z0,z0_estimation=meta)

def make_cycle(t,i,v,temp,net,z0,cfg):
    z=P.causal_soc(net,qnom_Ah=cfg.nominal_capacity_Ah,z0=z0)
    return {k:torch.tensor(a,dtype=torch.float32) for k,a in
            dict(t=t,I=i,V=v,T_cell=temp,Ah=net,SoC=z).items()}

def dataset_context(cycle,cfg,stats=None,source=False):
    wc=WindowConfig(length=cfg.window_length,stride=cfg.window_stride,normalize=False)
    ds=BatteryWindowDataset(cycle,wc)
    starts,tr,va=P.temporal_split(len(cycle['V']),cfg)
    ds.starts=starts.tolist()
    if stats is None:
        mask=np.zeros(len(cycle['V']),dtype=bool)
        for s in starts[tr]:mask[s:s+cfg.window_length]=True
        stats=compute_norm_stats(cycle['I'][mask],cycle['V'][mask],cycle['T_cell'][mask])
    ds.cfg=WindowConfig(length=cfg.window_length,stride=cfg.window_stride,normalize=True)
    ds.renormalize(stats)
    idx=va if source else np.arange(len(starts))
    mask=np.zeros(len(cycle['V']),dtype=bool)
    for s in starts[idx]:mask[s:s+cfg.window_length]=True
    return dict(cycle=cycle,ds=ds,tr=tr,eval_indices=idx,mask=mask,stats=stats,
                starts=starts,source=source)

def loader(ctx,cfg,indices,seed=42,shuffle=False):
    return DataLoader(Subset(ctx['ds'],list(map(int,indices))),batch_size=cfg.batch_size,
        shuffle=shuffle,generator=torch.Generator().manual_seed(seed),num_workers=0)

def rk4_rc(I,r,c,dt):
    x=-dt/(r*c);a=1+x+x*x/2+x**3/6+x**4/24
    return lfilter([0.,r*(1-a)],[1.,-a],I,axis=-1)

def nls_voltage(theta,I,Z,cfg,branches):
    coeff=theta[:6];r0=10.**theta[6];out=np.polynomial.chebyshev.chebval(2*np.clip(Z,0,1)-1,coeff)-r0*I
    for j in range(branches):
        r=10.**theta[7+j*2];tau=10.**theta[8+j*2]
        out-=rk4_rc(I,r,tau/r,cfg.dt_seconds)
    return out

def fit_nls(ctx,cfg,branches):
    ds=ctx['ds'];starts=ctx['starts'][ctx['tr']];L=cfg.window_length
    I=np.stack([-ds._I_raw[s:s+L].numpy() for s in starts]).astype(float)
    Z0=np.asarray([float(ds._SoC_raw[s]) for s in starts])
    # Internal SOC integration, exactly matching RK4 with held current and eta=1.
    Z=Z0[:,None]-np.c_[np.zeros(len(I)),np.cumsum(I[:,:-1],axis=1)]*cfg.dt_seconds/(cfg.nominal_capacity_Ah*3600.)
    V=np.stack([ds._V_raw[s:s+L].numpy() for s in starts]).astype(float)
    guess=np.r_[np.polynomial.chebyshev.chebfit(2*np.clip(Z,0,1).ravel()-1,V.ravel(),5),np.log10(.03)]
    for j in range(branches):guess=np.r_[guess,np.log10(.02),np.log10(5. if j==0 else 40.)]
    lo=np.r_[np.full(6,-10.),-3.,np.tile([-3.,np.log10(cfg.dt_seconds)],branches)]
    hi=np.r_[np.full(6,10.),np.log10(.5),np.tile([np.log10(.5),4.],branches)]
    weights=1e-3*np.array([0.]+[2.**k for k in range(1,6)])
    def fun(th):return np.r_[(nls_voltage(th,I,Z,cfg,branches)-V).ravel(),weights*th[:6]]
    start=time.perf_counter()
    result=least_squares(fun,np.clip(guess,lo+1e-6,hi-1e-6),bounds=(lo,hi),method='trf',
                         x_scale='jac',max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    if not result.success:raise RuntimeError(f'NLS did not converge: {result.message}')
    th=result.x;r0=10.**th[6];r1=10.**th[7];c1=10.**th[8]/r1
    kwargs=dict(ocv_coeffs=th[:6],R0=r0,R1=r1,C1=c1,
        soc_train_min=float(Z.min()),soc_train_max=float(Z.max()),
        identification_rmse_V=float(np.sqrt(np.mean(result.fun[:V.size]**2))),
        identification_n_windows=len(I),identification_success=True)
    if branches==1:params=ECMParams(**kwargs)
    else:params=ECMParams2RC(**kwargs,R2=10.**th[9],C2=10.**th[10]/10.**th[9])
    info=dict(theta=th.tolist(),params=params.to_dict(),optimizer='scipy TRF nonlinear least squares',
              nfev=result.nfev,message=result.message,elapsed_seconds=time.perf_counter()-start,
              internal_soc=True,eta=1.,solver='classical RK4',
              tau_bounds_s=[cfg.dt_seconds,1e4],bound_hits=np.flatnonzero(result.active_mask).tolist())
    print('NLS',branches,'RC',info,flush=True)
    return params,info

def nls_predict(params,cycle,cfg,starts=None):
    L=cfg.window_length if starts is not None else len(cycle['V'])
    starts=[0] if starts is None else starts
    I=np.stack([-cycle['I'][s:s+L].numpy() for s in starts]).astype(float)
    Z0=np.asarray([float(cycle['SoC'][s]) for s in starts])
    Z=Z0[:,None]-np.c_[np.zeros(len(I)),np.cumsum(I[:,:-1],axis=1)]*cfg.dt_seconds/(cfg.nominal_capacity_Ah*3600.)
    out=np.polynomial.chebyshev.chebval(2*np.clip(Z,0,1)-1,params.ocv_coeffs)-params.R0*I
    out-=rk4_rc(I,params.R1,params.C1,cfg.dt_seconds)
    if hasattr(params,'R2'):out-=rk4_rc(I,params.R2,params.C2,cfg.dt_seconds)
    return out

def reconstruct(preds,ctx):
    n=len(ctx['cycle']['V']);summed=np.zeros(n);count=np.zeros(n)
    for pred,idx in zip(preds,ctx['eval_indices']):
        s=ctx['starts'][idx];summed[s:s+len(pred)]+=pred;count[s:s+len(pred)]+=1
    assert np.array_equal(count>0,ctx['mask'])
    return summed[ctx['mask']]/count[ctx['mask']]

def build_model(name,stats,params,cfg):
    if name=='lstm':return LSTMBaseline(**P.LSTM)
    return UnifiedECMUDE(stats,params,cfg,{'physics_gradient':0,'output_residual':1,'ecm_ude':2}[name])

def evaluate(model,name,ctx,cfg):
    truth=ctx['cycle']['V'].numpy()[ctx['mask']]
    if name.endswith('_nls'):
        windows=nls_predict(model,ctx['cycle'],cfg,ctx['starts'][ctx['eval_indices']])
    else:
        pred,_=predict(model,loader(ctx,cfg,ctx['eval_indices']),device='cpu')
        windows=pred.numpy()*ctx['stats']['V_std']+ctx['stats']['V_mean']
    out={'windowed':P.metrics_mV(reconstruct(windows,ctx),truth)}
    if name!='lstm':
        if name.endswith('_nls'):continuous=nls_predict(model,ctx['cycle'],cfg)[0]
        else:
            with torch.no_grad():continuous=(model(ctx['ds'].X[None])[0].numpy()*ctx['stats']['V_std']+ctx['stats']['V_mean'])
        out['continuous']=P.metrics_mV(continuous[ctx['mask']],truth)
    return out

def source_metadata(ctx):
    ds=ctx['ds'];tr=ctx['tr'];va=ctx['eval_indices'];L=ds.cfg.length
    def range_for(indices):
        z=np.concatenate([ds._SoC_raw[ds.starts[i]:ds.starts[i]+L].numpy() for i in indices])
        return [float(z.min()),float(z.max())]
    return dict(n_samples=len(ds.V),train_windows=len(tr),heldout_windows=len(va),
        train_last_sample_exclusive=int(ds.starts[tr[-1]]+L),heldout_first_sample=int(ds.starts[va[0]]),
        evaluated_samples=int(ctx['mask'].sum()),train_soc_range=range_for(tr),heldout_soc_range=range_for(va),stats=ctx['stats'])

def provenance():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    sources={str(p.relative_to(ROOT)):P.sha256(p) for p in sorted(ROOT.glob('*.py'))}
    packages={}
    for name in ('torch','numpy','scipy','torchdiffeq','openpyxl','numba','llvmlite'):
        packages[name]=importlib.metadata.version(name)
    return dict(commit_hash=commit,branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        preservation_commit='389172b',legacy_results_backup='../backups/results_20261101_011942.zip',
        legacy_results_backup_sha256='36aadc83662ef1db09ae26ebe1c457f2ed08a3d2008a4d2514f496f28e117a1f',
        source_sha256=sources,packages=packages,python=sys.version,platform=platform.platform(),
        command=sys.argv,started_at=now(),protocol=P.manifest())

def run_dataset(dataset,summary):
    cfg=P.DATASETS[dataset];source='UDDS_25C' if dataset=='panasonic' else 'FUDS_25C'
    calibration,calmeta=ocv_calibration() if dataset=='calce' else (None,dict(available=False,reason='no independent Panasonic OCV calibration found locally'))
    names=list(P.PANASONIC_FILES if dataset=='panasonic' else CALCE_FILES)
    contexts={};data=summary['datasets'].setdefault(dataset,dict(conditions={},runs={},ocv_calibration=calmeta))
    for condition in names:
        cycle,meta=load_panasonic(condition,cfg) if dataset=='panasonic' else load_calce(condition,cfg,calibration)
        ctx=dataset_context(cycle,cfg,stats=contexts[source]['stats'] if condition!=source else None,source=condition==source)
        contexts[condition]=ctx
        meta.update(n_samples=len(cycle['V']),evaluated_samples=int(ctx['mask'].sum()),
                    evaluated_soc_range=[float(cycle['SoC'][ctx['mask']].min()),float(cycle['SoC'][ctx['mask']].max())])
        if condition==source:meta['split']=source_metadata(ctx)
        data['conditions'][condition]=meta
        print('DATA',dataset,condition,meta,flush=True)
    save_json(P.OUTPUT/'summary.json',summary)
    params_by_name={}
    for branches in (1,2):
        name=f'ecm_{branches}rc_nls';path=P.OUTPUT/'checkpoints'/dataset/f'{name}.json'
        with log_file(f'{dataset}_{name}'):
            if path.exists():
                info=json.loads(path.read_text());cls=ECMParams if branches==1 else ECMParams2RC
                params=cls.from_dict(info['params']);print('RESUME',path)
            else:
                params,info=fit_nls(contexts[source],cfg,branches);save_json(path,info)
            params_by_name[name]=params
            run=data['runs'].setdefault(name,{})
            if 'deterministic' not in run:
                metrics={c:evaluate(params,name,ctx,cfg) for c,ctx in contexts.items()}
                run['deterministic']=dict(checkpoint=str(path),sha256=P.sha256(path),identification=info,metrics=metrics,completed_at=now())
                save_json(P.OUTPUT/'summary.json',summary)
    for name in P.MODELS[2:]:
        for seed in P.SEEDS:
            run=data['runs'].setdefault(name,{})
            if str(seed) in run:print('SKIP complete',dataset,name,seed,flush=True);continue
            with log_file(f'{dataset}_{name}_seed{seed}'):
                set_seed(seed);ctx=contexts[source]
                model=build_model(name,ctx['stats'],params_by_name['ecm_1rc_nls'],cfg)
                ckpt=P.OUTPUT/'checkpoints'/dataset/f'{name}_seed{seed}.pt'
                histpath=ckpt.with_suffix('.history.json');start=time.perf_counter()
                if ckpt.exists() and histpath.exists():
                    model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True));history=json.loads(histpath.read_text());print('RESUME checkpoint',ckpt)
                else:
                    conf=TrainConfig(**(P.LSTM_TRAIN if name=='lstm' else P.TRAIN),ckpt_path=str(ckpt))
                    history=asdict(train(model,loader(ctx,cfg,ctx['tr'],seed,True),loader(ctx,cfg,ctx['eval_indices']),conf))
                    history['elapsed_seconds']=time.perf_counter()-start;save_json(histpath,history)
                metrics={}
                for c,test in contexts.items():
                    metrics[c]=evaluate(model,name,test,cfg)
                    print('METRICS',dataset,name,seed,c,metrics[c],flush=True)
                run[str(seed)]=dict(seed=seed,checkpoint=str(ckpt),sha256=P.sha256(ckpt),history=str(histpath),
                    n_params=sum(p.numel() for p in model.parameters()),best_epoch=history['best_epoch'],
                    physical_params=model.physical_params_dict() if name!='lstm' else None,
                    elapsed_seconds=time.perf_counter()-start,metrics=metrics,completed_at=now())
                save_json(P.OUTPUT/'summary.json',summary);write_tables(summary)

def values(data,model,condition,mode,metric):
    runs=data['runs'].get(model,{})
    if model.endswith('_nls'):
        if 'deterministic' not in runs:return None
        val=runs['deterministic']['metrics'][condition][mode][metric]
        return np.full(len(P.SEEDS),val)
    if mode=='continuous' and model=='lstm':return None
    if not all(str(s) in runs for s in P.SEEDS):return None
    return np.array([runs[str(s)]['metrics'][condition][mode][metric] for s in P.SEEDS])

def paired(a,b):
    delta=np.round(np.asarray(a)-np.asarray(b),10)
    if np.all(delta==0):return dict(statistic=0.,p_raw=1.,n=5,method='all paired differences zero')
    # Exhaust all sign flips: exact even with tied ranks and zero differences.
    nonzero=delta[delta!=0];ranks=rankdata(np.abs(nonzero));total=ranks.sum()
    wplus=ranks[nonzero>0].sum();stat=min(wplus,total-wplus)
    possible=[sum(rank for rank,sign in zip(ranks,signs) if sign)
              for signs in itertools.product((0,1),repeat=len(nonzero))]
    p=sum(min(w,total-w)<=stat+1e-12 for w in possible)/len(possible)
    return dict(statistic=float(stat),p_raw=float(p),n=5,n_nonzero=int(len(nonzero)),
                method='two-sided exact Wilcoxon signed-rank; exhaustive sign permutations, zero differences removed')

def holm(records):
    ordered=sorted(records,key=lambda r:r['p_raw']);prev=0.;m=len(ordered)
    for j,r in enumerate(ordered):
        prev=max(prev,min(1.,(m-j)*r['p_raw']));r['p_holm']=prev;r['significant_005']=prev<.05

def model_comparison(a,b,model_a,model_b):
    if model_a.endswith('_nls') and model_b.endswith('_nls'):
        return dict(statistic=None,p_raw=None,p_holm=None,n=0,significant_005=None,
                    method='not applicable: two deterministic fits have no seed replication')
    return paired(a,b)

def write_tables(summary):
    rows=[];tests=[];winners=[]
    for dataset,data in summary['datasets'].items():
        for condition in data['conditions']:
            for mode in ('windowed','continuous'):
                arrays={model:{metric:values(data,model,condition,mode,metric) for metric in P.METRICS} for model in P.MODELS}
                available=[m for m in P.MODELS if arrays[m]['mae'] is not None]
                for model in available:
                    row=dict(dataset=dataset,condition=condition,evaluation=mode,model=model,n_seeds=1 if model.endswith('_nls') else 5)
                    for metric in P.METRICS:
                        a=arrays[model][metric];row[metric+'_mean_mV']=float(a.mean());row[metric+'_std_mV']=float(a.std(ddof=1))
                    rows.append(row)
                for metric in P.METRICS:
                    family=[]
                    for a,b in itertools.combinations(available,2):
                        r=dict(dataset=dataset,condition=condition,evaluation=mode,metric=metric,model_a=a,model_b=b,**model_comparison(arrays[a][metric],arrays[b][metric],a,b));family.append(r)
                    holm([r for r in family if r['p_raw'] is not None]);tests.extend(family)
                    ranked=sorted(available,key=lambda m:arrays[m][metric].mean())
                    if len(ranked)>1:
                        a,b=ranked[:2];test=next(r for r in family if {r['model_a'],r['model_b']}=={a,b})
                        winners.append(dict(dataset=dataset,condition=condition,evaluation=mode,metric=metric,winner=a,runner_up=b,
                            winner_mean_mV=float(arrays[a][metric].mean()),runner_up_mean_mV=float(arrays[b][metric].mean()),
                            p_raw=test['p_raw'],p_holm=test['p_holm'],significant_005=test['significant_005']))
    if rows:
        with (P.OUTPUT/'table_main.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        lines=[r'\begin{longtable}{llllrrrr}',r'\caption{Unified protocol. Errors in mV, mean $\pm$ sample SD; NLS deterministic.}\\',
               r'Dataset & Condition & Mode & Model & MAE & RMSE & Max & P99 \\',r'\hline']
        for row in rows:
            labels=[str(row[k]).replace('_',r'\_') for k in ('dataset','condition','evaluation','model')]
            nums=[f"{row[m+'_mean_mV']:.3f} $\\pm$ {row[m+'_std_mV']:.3f}" for m in P.METRICS]
            lines.append(' & '.join(labels+nums)+r' \\')
        lines.append(r'\end{longtable}');(P.OUTPUT/'table_main.tex').write_text('\n'.join(lines)+'\n')
    save_json(P.OUTPUT/'statistics.json',dict(primary_metric='mae',unit='seed',alpha=.05,
        note='With n=5 nonzero pairs, the minimum exact two-sided p is 0.0625. No temporal pseudoreplication. NLS is deterministic, not five independent fits; NLS versus NLS has no inferential test and is excluded from the Holm family.',
        all_pairwise=tests,ude_vs_baselines=[r for r in tests if 'ecm_ude' in (r['model_a'],r['model_b'])],winners=winners))
    if winners:
        with (P.OUTPUT/'winners.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(winners[0]));w.writeheader();w.writerows(winners)

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['all','panasonic','calce'],default='all')
    parser.add_argument('--report-only',action='store_true',help='Regenerate tables/statistics from saved per-seed metrics, without loading data or fitting models')
    args=parser.parse_args(argv);torch.set_num_threads(1)
    P.OUTPUT.mkdir(parents=True,exist_ok=True)
    current=provenance();path=P.OUTPUT/'summary.json'
    if args.report_only:
        summary=json.loads(path.read_text())
        with log_file('reporting'):
            write_tables(summary)
            summary['report_provenance']={**current,'generated_at':now(),
                'scope':'tables/statistics only; original training manifest and metrics unchanged'}
            ledger=P.OUTPUT/'parallel_preparations.json'
            if ledger.exists():summary['manifest']['parallel_preparations']=json.loads(ledger.read_text())
            save_json(path,summary)
        return
    if path.exists():
        summary=json.loads(path.read_text())
        for key in ('source_sha256','protocol'):
            if summary['manifest'][key]!=current[key]:raise RuntimeError(f'Cannot mix protocols/code in resumed run: changed {key}')
    else:summary=dict(manifest=current,status='running',datasets={})
    summary['status']='running';save_json(path,summary)
    with log_file('execution'):
        try:
            for dataset in (['panasonic','calce'] if args.dataset=='all' else [args.dataset]):run_dataset(dataset,summary)
            summary['status']='complete' if len(summary['datasets'])==2 else 'dataset_complete'
            summary['completed_at']=now();write_tables(summary);save_json(path,summary)
        except BaseException as exc:
            summary['status']='failed';summary['error']=repr(exc);summary['failed_at']=now();save_json(path,summary);raise

if __name__=='__main__':main()
