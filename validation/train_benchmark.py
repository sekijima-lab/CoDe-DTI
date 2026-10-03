import sys,argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('repo');p.add_argument('out');p.add_argument('--modern',action='store_true');a=p.parse_args();sys.path.insert(0,str(Path(a.repo).resolve()))
from bremen.albrecht import Albrecht
if a.modern:
 import torch
 torch.set_num_threads(1)
folder=Path('code-dti-runtime-validation/fixtures');folder.mkdir(exist_ok=True)
import gzip,json
with gzip.open(Path(a.repo,'intmat.npy.gz'),'rb') as fp_in:
 full_matrix=np.load(fp_in,allow_pickle=False)
targets=np.argsort(-full_matrix.sum(axis=0),kind='stable')[:12]
drugs=np.argsort(-full_matrix[:,targets].sum(axis=1),kind='stable')[:120]
all_lines=Path(a.repo,'fingerprints.txt').read_text().splitlines()
fp=folder/'fingerprints.txt';fp.write_text('\n'.join(all_lines[i] for i in drugs)+'\n')
im=full_matrix[np.ix_(drugs,targets)].astype('f')
np.save(folder/'intmat.npy',im)
(folder/'indices.json').write_text(json.dumps({'drug_indices':drugs.tolist(),'target_indices':targets.tolist(),'positive_count':int(im.sum())},indent=2)+'\n')
out={}
for seed in (0,19,73):
 for kind in ('bremen','wang'):
  np.random.seed(seed);model=Albrecht(8,120,12,str(fp),1024,2,encoder=kind,n_layer=4,lambda_w=.5,lambda_v=.5,lambda_n=4.,lambda_u=1.,alpha=4.)
  model.fit(im)
  q=f'{seed}_{kind}_'
  out[q+'theta']=model.ctr.theta.copy();out[q+'u']=model.ctr.u.copy();out[q+'v']=model.ctr.v.copy();out[q+'predict']=model.predict()
  params=model.encoder.named_legacy_parameters() if a.modern else sorted(model.encoder.namedparams())
  for name,param in params:out[q+name]=param.detach().numpy().copy() if a.modern else param.data.copy()
np.savez_compressed(a.out,**out);print('TRAIN_BENCHMARK_PASS',len(out))
