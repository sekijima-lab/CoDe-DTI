import sys,argparse
from pathlib import Path
import numpy as np
from rdkit import Chem
p=argparse.ArgumentParser();p.add_argument('repo');p.add_argument('out');p.add_argument('--modern',action='store_true');a=p.parse_args();sys.path.insert(0,str(Path(a.repo).resolve()))
from bremen.fp import nfp
if a.modern:
 import torch
 torch.set_num_threads(1)
output={}
for seed in (0,19,73):
 np.random.seed(seed);model=nfp(8,4,2)
 for smiles in ('CCO','c1ccccc1','CC(=O)O'):
  value=model.get_nfp(Chem.MolFromSmiles(smiles),train=True)
  output[f'{seed}_{smiles}']=value.detach().numpy() if a.modern else value.data
np.savez_compressed(a.out,**output);print('NFP_PASS')
