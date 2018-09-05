from bremen.naive_mf import NaiveMFDriver
from bremen.cross_validation import cross_validation, CrossValidation
from gzip import GzipFile
import json
import os
import itertools
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

def grid_iter(params: dict):
    names = tuple(params.keys())
    for values in itertools.product(*list(params.values())):
        yield dict(zip(names, values))

def main():
    params_candidates = {
        'lambda_u': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        'lambda_v': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        'alpha': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        'n_latent': [50, 100, 200, 300],
    }
    with Pool(processes=4) as p:
        results = p.map(job, enumerate(grid_iter(params_candidates)))

    with open('naive_mf/results.json', 'w') as fp:
        json.dump(results, fp)


def job(args):
    i, params = args
    dir_name = 'naive_mf/iter_%d' % i
    os.makedirs(dir_name, 0o755, True)
    with open(os.path.join(dir_name, 'params.json'), 'w') as fp:
        json.dump(params, fp)
    result = evaluate(1, epoch=50, n_fold=5, intmat_path='../intmat.npy.gz', output_dir=dir_name, **params)
    with open(os.path.join(dir_name, 'result.json'), 'w') as fp:
        json.dump(result, fp)
    return result


def evaluate(cv_mode, alpha, epoch, n_latent, lambda_v, lambda_u, intmat_path, n_layer=4, n_fold=5, output_dir='.'):
    parameters = {
        'lambda_u': lambda_u,
        'lambda_v': lambda_v,
        'alpha': alpha,
    }

    if intmat_path.endswith('.npy'):
        intmat = np.load(intmat_path)
    elif intmat_path.endswith('.npy.gz'):
        with GzipFile(intmat_path, 'r') as gf:
            intmat = np.load(gf)
    else:
        intmat = np.loadtxt(intmat_path)

    cvsmode = cv_mode
    n_drug = intmat.shape[0]
    n_target = intmat.shape[1]
    albrecht = lambda: NaiveMFDriver(n_latent=n_latent, n_drug=n_drug, n_target=n_target, epoch=epoch, **parameters)
    result = CrossValidationAvg(np.random.RandomState(0), output_dir) \
        .cross_validation(albrecht, intmat, n_drug, n_target, n_fold, cvsmode)
    return result


class CrossValidationAvg(CrossValidation):
    def __init__(self, random_state, output_dir):
        super().__init__(random_state, output_dir)
        self.tprs = []
        self.mean_fpr = np.linspace(0, 1, 10000)

    def after_fit(self):
        super().after_fit()

    def prepare(self):
        super().prepare()

    def after_test(self, fold, true_tests, predicted_tests):
        from sklearn.metrics import roc_curve
        super().after_test(fold, true_tests, predicted_tests)

        fpr, tpr, _ = roc_curve(true_tests, predicted_tests, drop_intermediate=False)
        self.tprs.append(np.interp(self.mean_fpr, fpr, tpr))

    def before_fit(self):
        super().before_fit()

    def after_all(self):
        tprs = np.array(self.tprs)
        avg_tpr = tprs.mean(axis=0)
        with GzipFile(os.path.join(self.output_dir, 'avg_tpr.npy.gz'), 'w') as fp:
            np.save(fp, avg_tpr)
        return super().after_all()


if __name__ == '__main__':
    main()
