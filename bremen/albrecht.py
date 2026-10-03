from .ctr import CTR
from .cdl import FingerprintDataset, CDLUpdater
from .autoencoder import AutoEncoder, WangAutoEncoder
from . import cross_validation
import torch
from .cdl import SerialIterator
from gzip import GzipFile

import numpy as np
from scipy.sparse import coo_matrix
import json
import sys
import os


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output-dir', default='.')
    p.add_argument('--n-fold', type=int, default=5)
    p.add_argument('--n-layer', type=int, default=4)
    p.add_argument('--encoder', choices=['bremen', 'wang'], default='bremen')
    p.add_argument('--n-latent', type=int, default=50)
    p.add_argument('--fingerprint-bit', type=int, default=1024)
    p.add_argument('interaction_matrix', type=str)
    p.add_argument('fingerprint_file', type=str)
    p.add_argument('--lambda-v', '-lv', type=float, default=1)
    p.add_argument('--lambda-u', '-lu', type=float, default=10)
    p.add_argument('--lambda-w', '-lw', type=float, default=1)
    p.add_argument('--lambda-n', '-ln', type=float, default=100)
    p.add_argument('--alpha', type=float, default=40.0)
    p.add_argument('--epoch', type=int, default=10)
    p.add_argument('--cv-mode', type=int, default=1, choices=[1, 2, 3])
    p.add_argument('--result-bank', type=str)
    args = p.parse_args()

    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'runtime.json'), 'w') as fp:
        json.dump({'python': sys.version.split()[0], 'torch': str(torch.__version__),
                   'numpy': np.__version__, 'device': 'cpu', 'threads': 1,
                   'seed': args.seed, 'compatibility': 'historical CPU sigmoid/RNG/iterator'}, fp, indent=2)
    result = evaluate(args.cv_mode, args.alpha, args.epoch, args.n_latent, args.lambda_n,
                      args.lambda_w, args.lambda_v, args.lambda_u, args.fingerprint_bit,
                      args.interaction_matrix, args.fingerprint_file, encoder=args.encoder, n_layer=args.n_layer, n_fold=args.n_fold, output_dir=args.output_dir)

    if args.result_bank:
        bank = []
        if os.path.exists(args.result_bank):
            with open(args.result_bank) as fp:
                bank = json.load(fp)
        if not isinstance(bank, list):
            print(json.dumps(result))
            return 1
        bank.append({
            'alpha': args.alpha,
            'epoch': args.epoch,
            'n_latent': args.n_latent,
            'lambda_n': args.lambda_n,
            'lambda_w': args.lambda_w,
            'lambda_v': args.lambda_v,
            'lambda_u': args.lambda_u,
            'result_metrics': result,
        })
        with open(args.result_bank, 'w') as fp:
            json.dump(bank, fp)
        return 0
    else:
        print(json.dumps(result))
        return 0

def evaluate(cv_mode, alpha, epoch, n_latent, lambda_n, lambda_w, lambda_v, lambda_u,
             fingerprint_bit, intmat_path, fingerprint_path, encoder='bremen', n_layer=4, n_fold=5, output_dir='.'):
    cdl_parameters = {
        'lambda_n': lambda_n,
        'lambda_u': lambda_u,
        'lambda_v': lambda_v,
        'lambda_w': lambda_w,
        'alpha': alpha,
    }

    if intmat_path.endswith('.npy'):
        intmat = np.load(intmat_path, allow_pickle=False)
    elif intmat_path.endswith('.npy.gz'):
        with GzipFile(intmat_path, 'r') as gf:
            intmat = np.load(gf, allow_pickle=False)
    else:
        intmat = np.loadtxt(intmat_path)

    cvsmode = cv_mode
    n_drug = intmat.shape[0]
    n_target = intmat.shape[1]
    albrecht = lambda: Albrecht(n_latent, n_drug, n_target, fingerprint_path,
                                fingerprint_bit, epoch, encoder, n_layer, None, **cdl_parameters)
    result = cross_validation.cross_validation(np.random.RandomState(0), albrecht,
                                               intmat, n_drug, n_target, n_fold, cvsmode, output_dir=output_dir)
    return result


class Albrecht:
    def __init__(self, n_latent, n_drug, n_target, fingerprint_file, fingerprint_bit, epoch,
                 encoder='bremen', n_layer=4,
                 device=None, **cdl_parameters):
        self.ctr = CTR(n_latent, n_target, n_drug, device=device, **cdl_parameters)
        if encoder == 'bremen':
            self.encoder = AutoEncoder(fingerprint_bit, n_latent, n_layer)
        elif encoder == 'wang':
            self.encoder = WangAutoEncoder(fingerprint_bit, n_latent, n_layer)
        self.dataset = FingerprintDataset(fingerprint_file)
        self.stop_epoch = epoch
        self.cdl_parameters = cdl_parameters

    def fit(self, intmat):
        coo = coo_matrix(intmat)
        logs = [(target, drug, data) for target, drug, data in zip(coo.col, coo.row, coo.data)]
        it = SerialIterator(self.dataset, 100)
        opt = torch.optim.SGD(self.encoder.parameters(), lr=0.001)
        updater = CDLUpdater(self.ctr, self.encoder, logs, it, opt, None, **self.cdl_parameters)
        self.training_log = []
        while it.epoch < self.stop_epoch:
            metrics = updater.update_core()
            self.training_log.append(metrics)
        self.encoder.eval()

    def predict(self):
        u = self.ctr.u
        v = self.ctr.v
        return np.matmul(u, v.T).T


if __name__ == '__main__':
    main()
