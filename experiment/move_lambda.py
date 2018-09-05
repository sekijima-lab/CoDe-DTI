from bremen.albrecht import Albrecht
from bremen.cross_validation import cross_validation, CrossValidation
from gzip import GzipFile
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def main():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    roc_aucs = {}

    params = {
        0: '$\lambda_v=0$ (naïve MF)',
        0.1: '$\lambda_v=0.1$',
        0.5: '$\lambda_v=0.5$',
        1.0: '$\lambda_v=1$',
        2.0: '$\lambda_v=2$',
        4.0: '$\lambda_v=4$',
        8.0: '$\lambda_v=8$',
        16.0: '$\lambda_v=16$',
        32.0: '$\lambda_v=32$',
        64: '$\lambda_v=64$',
    }
    for lambda_v, legend in params.items():
        dir_name = 'lambda_%g' % lambda_v
        if os.path.exists(os.path.join(dir_name, 'mean_tpr.npy.gz')):
            with GzipFile(os.path.join(dir_name, 'mean_tpr.npy.gz')) as fp:
                mean_tpr = np.load(fp)
            fpr = np.linspace(0, 1, 10000)
            ax.plot(fpr, mean_tpr, lw=2, label=legend)
            roc_aucs[lambda_v] = auc(fpr, mean_tpr)
        else:
            os.makedirs(dir_name, 0o755, True)
            result = evaluate(ax, legend, 1,
                              alpha=4, epoch=20, n_latent=100, lambda_n=32, lambda_w=0.5, lambda_u=32, lambda_v=lambda_v,
                              fingerprint_bit=1024, intmat_path='../intmat.npy.gz', fingerprint_path='../fingerprints.txt',
                              output_dir=dir_name, n_fold=5)

            roc_aucs[lambda_v] = result['avg_roc_auc']

        ax.legend(loc='lower right')
        fig.savefig('average_roc.eps')

    print(roc_aucs)


def evaluate(ax, legend, cv_mode, alpha, epoch, n_latent, lambda_n, lambda_w, lambda_v, lambda_u,
             fingerprint_bit, intmat_path, fingerprint_path, encoder='bremen', n_layer=4, n_fold=5, output_dir='.'):
    cdl_parameters = {
        'lambda_n': lambda_n,
        'lambda_u': lambda_u,
        'lambda_v': lambda_v,
        'lambda_w': lambda_w,
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
    albrecht = lambda: Albrecht(n_latent, n_drug, n_target, fingerprint_path,
                                fingerprint_bit, epoch, encoder, n_layer, None, **cdl_parameters)
    result = CrossValidationAvg(np.random.RandomState(0), output_dir, ax, legend) \
        .cross_validation(albrecht, intmat, n_drug, n_target, n_fold, cvsmode)
    return result


class CrossValidationAvg(CrossValidation):
    def __init__(self, random_state, output_dir, avg_ax, legend):
        super().__init__(random_state, output_dir)
        self.avg_ax = avg_ax
        self.tprs = []
        self.mean_fpr = np.linspace(0, 1, 10000)
        self.legend = legend

    def after_test(self, fold, true_tests, predicted_tests):
        super().after_test(fold, true_tests, predicted_tests)

        fpr, tpr, _ = roc_curve(true_tests, predicted_tests, drop_intermediate=False)
        self.tprs.append(np.interp(self.mean_fpr, fpr, tpr))

    def after_all(self):
        tprs = np.array(self.tprs)
        avg_tpr = tprs.mean(axis=0)
        with GzipFile(os.path.join(self.output_dir, 'mean_tpr.npy.gz'), 'w') as fp:
            np.save(fp, avg_tpr)
        self.avg_ax.plot(self.mean_fpr, avg_tpr, lw=2, label=self.legend)
        return super().after_all()


if __name__ == '__main__':
    main()
