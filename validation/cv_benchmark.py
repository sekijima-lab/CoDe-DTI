import argparse,sys,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('repo');p.add_argument('out');p.add_argument('--modern',action='store_true');a=p.parse_args();sys.path.insert(0,str(Path(a.repo).resolve()))
from bremen.albrecht import evaluate
if a.modern:
 import torch
 torch.set_num_threads(1)
np.random.seed(19)
output=Path(a.out).with_suffix('');output.mkdir(exist_ok=True)
result=evaluate(1,4,1,8,4,.5,.5,1,1024,'code-dti-runtime-validation/fixtures/intmat.npy','code-dti-runtime-validation/fixtures/fingerprints.txt',encoder='wang',n_layer=4,n_fold=2,output_dir=str(output))
result={k:float(v) for k,v in result.items() if 'elapsed' not in k}
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(result)
