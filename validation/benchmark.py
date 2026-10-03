import argparse,sys,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('repo');p.add_argument('out');p.add_argument('--modern',action='store_true');a=p.parse_args();sys.path.insert(0,str(Path(a.repo).resolve()))
from bremen.autoencoder import AutoEncoder,WangAutoEncoder
from bremen.ctr import CTR
from bremen.cross_validation import cross_validation_divisions
if a.modern:
 import torch
 torch.set_num_threads(1)
else:
 import chainer
 import chainer.functions as F
 import chainer.optimizers as O
out={}
for seed in (0,19,73):
 for kind,cls in [('bremen',AutoEncoder),('wang',WangAutoEncoder)]:
  np.random.seed(seed);model=cls(1024,8,4)
  prefix=f'{seed}_{kind}_'
  if a.modern: params=list(model.named_legacy_parameters())
  else: params=[(name,param) for name,param in sorted(model.namedparams())]
  for name,param in params: out[prefix+'init'+name]=param.detach().numpy().copy() if a.modern else param.data.copy()
  lines=Path(a.repo,'fingerprints.txt').read_text().splitlines()[:8]
  x=np.asarray([list(map(float,line.split()[1])) for line in lines],dtype='f')
  out[prefix+'x']=x
  for train in (False,True):
   np.random.seed(seed+101)
   if a.modern:
    model.zero_grad();code,recon=model(torch.from_numpy(x),test=not train)
    loss=(code.square().sum()+((torch.from_numpy(x)-recon).square().sum())+model.w_loss())/len(x)
    loss.backward(); val=lambda t:t.detach().numpy().copy()
   else:
    model.cleargrads();code,recon=model(x,test=not train)
    loss=(F.sum(F.square(code))+F.sum(F.square(x-recon))+model.w_loss())/len(x)
    loss.backward(); val=lambda t:t.data.copy()
   q=prefix+('train_' if train else 'eval_')
   out[q+'code']=val(code);out[q+'recon']=val(recon);out[q+'loss']=val(loss)
   for name,param in params:out[q+'grad'+name]=param.grad.detach().numpy().copy() if a.modern else param.grad.copy()
  if a.modern:torch.optim.SGD(model.parameters(),lr=.001).step()
  else:
   opt=O.SGD(.001);opt.setup(model);opt.update()
  for name,param in params:out[prefix+'sgd'+name]=param.detach().numpy().copy() if a.modern else param.data.copy()
 np.random.seed(seed);ctr=CTR(8,6,8);logs=[(j,i,1) for i in range(8) for j in range(6) if (i+j)%3==0];ctr.als(logs)
 out[f'{seed}_als_u']=ctr.u.copy();out[f'{seed}_als_v']=ctr.v.copy();out[f'{seed}_pred']=np.matmul(ctr.u,ctr.v.T).T
 for mode in (1,2,3):
  for i,fold in enumerate(cross_validation_divisions(np.random.RandomState(seed),8,6,3,mode)):out[f'{seed}_cv{mode}_{i}']=fold
np.savez_compressed(a.out,**out);print('BENCHMARK_PASS',len(out))
