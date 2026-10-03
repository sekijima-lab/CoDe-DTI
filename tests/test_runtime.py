import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from bremen.autoencoder import AutoEncoder,WangAutoEncoder,DL
from bremen.ctr import CTR
from bremen.cdl import FingerprintDataset,SerialIterator
from bremen.fp import nfp
from rdkit import Chem

class RuntimeTests(unittest.TestCase):
 def test_encoder_reference_forward_and_gradients(self):
  torch.set_num_threads(1)
  ref=np.load(Path(__file__).resolve().parents[1]/'validation/reference.npz',allow_pickle=False);self.addCleanup(ref.close)
  for kind,cls in [('bremen',AutoEncoder),('wang',WangAutoEncoder)]:
   np.random.seed(0);model=cls(1024,8,4);q='0_'+kind+'_';x=torch.from_numpy(ref[q+'x'])
   for train in (False,True):
    np.random.seed(101);model.zero_grad();code,recon=model(x,test=not train);key=q+('train_' if train else 'eval_')
    np.testing.assert_allclose(code.detach().numpy(),ref[key+'code'],atol=1e-5,rtol=0)
    np.testing.assert_allclose(recon.detach().numpy(),ref[key+'recon'],atol=1e-5,rtol=0)
    loss=(code.square().sum()+(x-recon).square().sum()+model.w_loss())/len(x);loss.backward()
    self.assertLessEqual(abs(float(loss.detach())-float(ref[key+'loss'])),1e-4)
    for name,param in model.named_legacy_parameters():
     expected=ref[key+'grad'+name];actual=param.grad.numpy()
     if key+'grad'+name+'__indices' in ref:actual=actual.reshape(-1)[ref[key+'grad'+name+'__indices']]
     np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=0)
 def test_zero_dropout_still_consumes_historical_rng(self):
  np.random.seed(2);layer=DL(0,3,2);before=np.random.get_state();layer(torch.ones(4,3));after=np.random.rand()
  np.random.set_state(before);np.random.rand(4,2);self.assertEqual(after,np.random.rand())
 def test_iterator_wrap_and_shuffle(self):
  data=[(i,np.ones(4,dtype='f')) for i in range(120)];np.random.seed(19);iterator=SerialIterator(data);order=iterator.order.copy();tail=order[100:].tolist();state=np.random.get_state()
  first,epoch=iterator.next();self.assertFalse(epoch);self.assertEqual([x[0] for x in first],order[:100].tolist())
  second,epoch=iterator.next();self.assertTrue(epoch);np.random.set_state(state);np.random.shuffle(order)
  self.assertEqual([x[0] for x in second],tail+order[:80].tolist())
  self.assertEqual(iterator.position,80)
 def test_als_reference_and_safe_roundtrip(self):
  ref=np.load(Path(__file__).resolve().parents[1]/'validation/reference.npz',allow_pickle=False);self.addCleanup(ref.close);np.random.seed(0);ctr=CTR(8,6,8);ctr.als([(j,i,1) for i in range(8) for j in range(6) if (i+j)%3==0])
  np.testing.assert_allclose(ctr.u,ref['0_als_u'],atol=1e-4,rtol=0);np.testing.assert_allclose(ctr.v,ref['0_als_v'],atol=1e-4,rtol=0)
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'state.npz';ctr.save_npz(path);copy=CTR(8,6,8);copy.load_npz(path);np.testing.assert_array_equal(copy.v,ctr.v)
   np.savez(path,v=np.asarray([object()],dtype=object),u=ctr.u,theta=ctr.theta)
   with self.assertRaises(ValueError):copy.load_npz(path)
 def test_neural_fingerprint_and_gradients_finite(self):
  np.random.seed(19);model=nfp(8,4,2);fp=model.get_nfp(Chem.MolFromSmiles('CCO'));self.assertEqual(tuple(fp.shape),(1,8));self.assertAlmostEqual(float(fp.detach().sum()),6,places=5);fp.square().sum().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.model.parameters() if p.grad is not None))
 def test_non_cpu_and_empty_data_rejected(self):
  with self.assertRaises(ValueError):CTR(2,3,3,device=0)
  with self.assertRaises(ValueError):AutoEncoder(16,4,3)
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'fp';path.write_text('')
   with self.assertRaises(ValueError):FingerprintDataset(path)
if __name__=='__main__':unittest.main()
