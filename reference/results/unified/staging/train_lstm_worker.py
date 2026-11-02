"""Prepare a canonical LSTM seed without writing the main summary/checkpoints."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import time,json,subprocess
from dataclasses import asdict
import torch
import protocol as P
from exp_unified import load_panasonic,dataset_context,build_model,loader,log_file,save_json
from training import train,TrainConfig
from utils import set_seed
seed=int(sys.argv[1]);torch.set_num_threads(1)
original=P.OUTPUT
P.OUTPUT=original/'staging'/f'seed{seed}'
cfg=P.DATASETS['panasonic']
with log_file(f'panasonic_lstm_seed{seed}'):
 cycle,_=load_panasonic('UDDS_25C',cfg)
 ctx=dataset_context(cycle,cfg,source=True)
 set_seed(seed);model=build_model('lstm',ctx['stats'],None,cfg)
 ckpt=P.OUTPUT/'checkpoints'/f'lstm_seed{seed}.pt'
 t=time.perf_counter()
 history=asdict(train(model,loader(ctx,cfg,ctx['tr'],seed,True),loader(ctx,cfg,ctx['eval_indices']),TrainConfig(**P.LSTM_TRAIN,ckpt_path=str(ckpt))))
 history['elapsed_seconds']=time.perf_counter()-t
 save_json(ckpt.with_suffix('.history.json'),history)
 save_json(P.OUTPUT/'worker.json',dict(seed=seed,commit_hash=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),worker_sha256=P.sha256(__file__),checkpoint_sha256=P.sha256(ckpt),status='complete',source='panasonic/UDDS_25C',model='lstm',train_config=P.LSTM_TRAIN))
 print('WORKER READY',seed,flush=True)
