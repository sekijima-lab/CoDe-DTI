"""Compare full workspace arrays or compact committed baselines."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('updated');p.add_argument('--training',action='store_true');a=p.parse_args()
old=np.load(a.reference,allow_pickle=False);new=np.load(a.updated,allow_pickle=False)
limits=json.loads(Path(__file__).with_name('criteria.json').read_text())
maxima={};failed=[]
for key in old.files:
 if key.endswith('__indices'):continue
 actual=new[key]
 if key+'__indices' in old.files:actual=actual.reshape(-1)[old[key+'__indices']]
 expected=old[key]
 if expected.shape!=actual.shape:raise ValueError(f'Shape mismatch: {key}')
 diff=float(np.max(np.abs(expected-actual)))
 if a.training:category='prediction_max_abs' if key.endswith('_predict') else 'als_weights_max_abs' if key.endswith(('_theta','_u','_v')) else 'sgd_weights_max_abs'
 else:category='exact' if 'init' in key or '_cv' in key or key.endswith('_x') else 'gradient_max_abs' if 'grad' in key else 'sgd_weights_max_abs' if 'sgd' in key else 'loss_max_abs' if 'loss' in key else 'als_weights_max_abs' if '_als' in key else 'forward_max_abs'
 limit=0 if category=='exact' else limits[category]
 maxima[category]=max(maxima.get(category,0),diff)
 if diff>limit:failed.append({'key':key,'difference':diff,'limit':limit})
print(json.dumps({'verdict':'FAIL' if failed else 'PASS','scope':'compact samples if indexed; otherwise full arrays','maxima':maxima,'failed':failed},indent=2));raise SystemExit(bool(failed))
