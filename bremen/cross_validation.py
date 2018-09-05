import numpy as np
import math
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from gzip import GzipFile
import matplotlib.pyplot as plt
import os
import time


def cross_validation_divisions(rand: np.random.RandomState, n_drug, n_target, n_fold, validation_type):
    if validation_type == 1:
        return _cv1(rand, n_drug, n_target, n_fold)
    elif validation_type == 2:
        return _cv2(rand, n_drug, n_target, n_fold)
    elif validation_type == 3:
        return _cv3(rand, n_drug, n_target, n_fold)
    else:
        raise ValueError()


def _cv1(rand, n_drug, n_target, n_fold):
    x = np.asarray([[i // n_target, i % n_target] for i in range(n_drug * n_target)], dtype=np.int64)
    rand.shuffle(x)
    size = x.shape[0] // n_fold
    return [x[i * size:(i + 1) * size] for i in range(n_fold)]


def _cv2(rand, n_drug, n_target, n_fold):
    x = np.arange(n_drug)
    rand.shuffle(x)
    size = int(math.ceil(n_drug / n_fold))
    xx = list(x[i:i + size] for i in range(0, n_drug, size))
    return [np.asarray([[bb, i] for bb in b for i in range(n_target)], dtype=np.int64) for b in xx]


def _cv3(rand, n_drug, n_target, n_fold):
    retval = _cv2(rand, n_target, n_drug, n_fold)
    for r in retval:
        r[:, [0, 1]] = r[:, [1, 0]]
    return retval

def cross_validation(rand: np.random.RandomState, model_producer, interaction_matrix: np.ndarray,
                     n_drug, n_target, n_fold, validation_type, output_dir='.'):
    return CrossValidation(rand, output_dir)\
        .cross_validation(model_producer, interaction_matrix, n_drug, n_target, n_fold, validation_type)

class CrossValidation:
    def __init__(self, random_state, output_dir):
        self.random_state = random_state
        self.output_dir = output_dir

    def prepare(self):
        self.elapsed_times = []
        self.pr_curves_fig = plt.figure()
        self.pr_curves_ax = self.pr_curves_fig.add_subplot(1, 1, 1)
        self.pr_curves_ax.set_xlabel('Recall')
        self.pr_curves_ax.set_xlabel('Precision')
        self.pr_curves_ax.set_xlim([0.0, 1.0])
        self.pr_curves_ax.set_ylim([0.0, 1.0])

        self.roc_curves_fig = plt.figure()
        self.roc_curves_ax = self.roc_curves_fig.add_subplot(1, 1, 1)
        self.roc_curves_ax.set_xlabel('False Positive Rate')
        self.roc_curves_ax.set_ylabel('True Positive Rate')
        self.roc_curves_ax.set_xlim([0.0, 1.0])
        self.roc_curves_ax.set_ylim([0.0, 1.0])
        self.roc_curves_ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        self.tprs = []
        self.mean_fpr = np.linspace(0, 1, 10000)
        self.rocauc = []
        self.prauc = []
        self.rocauc_0_1 = []

    def after_test(self, fold, true_tests, predicted_tests):
        mask = true_tests == 1
        predicted_tests_masked = predicted_tests[mask]
        predicted_tests_negatives = predicted_tests[np.logical_not(mask)]

        plt.figure()
        try:
            # plt.plot(percentile_masked, predicted_tests_masked, 'x')
            plt.hist([predicted_tests_masked, predicted_tests_negatives],
                     normed=True, bins=30, color=['blue', 'red'], label=['Positives', 'Negatives'])
            plt.savefig(os.path.join(self.output_dir, 'probability_distribution_fold_%d.eps' % fold))
        except:
            pass
        finally:
            plt.close()

        try:
            precision, recall, _ = precision_recall_curve(true_tests, predicted_tests)

            pr_auc = auc(recall, precision)
            self.pr_curves_ax.plot(recall, precision, lw=2, label="fold:%d (AUC:%.4f)" % (fold, pr_auc))

            self.prauc.append(pr_auc)
        except:
            self.prauc.append(0.0)

        try:
            fpr, tpr, _ = roc_curve(true_tests, predicted_tests)

            roc_auc = auc(fpr, tpr)
            self.roc_curves_ax.plot(fpr, tpr, lw=2, label="fold:%d (AUC:%.4f)" % (fold, roc_auc))

            tpr_ = tpr.copy()
            tpr_[fpr > 0.1] = 0
            roc_auc_0_1 = auc(fpr, tpr)

            self.rocauc.append(roc_auc)
            self.rocauc_0_1.append(roc_auc_0_1)
        except:
            self.rocauc.append(0.5)
            self.rocauc_0_1.append(0.005)

        fpr, tpr, _ = roc_curve(true_tests, predicted_tests, drop_intermediate=False)
        self.tprs.append(np.interp(self.mean_fpr, fpr, tpr))


    def after_all(self):
        tprs = np.array(self.tprs)
        avg_tpr = tprs.mean(axis=0)
        with GzipFile(os.path.join(self.output_dir, 'avg_tpr.npy.gz'), 'w') as fp:
            np.save(fp, avg_tpr)
        self.pr_curves_ax.legend(loc="upper right")
        self.roc_curves_ax.legend(loc="lower right")

        self.pr_curves_fig.savefig(os.path.join(self.output_dir, 'pr_curve.eps'))
        self.roc_curves_fig.savefig(os.path.join(self.output_dir, 'roc_curve.eps'))

        plt.close(self.pr_curves_fig)
        plt.close(self.roc_curves_fig)

        return {
            'avg_roc_auc': np.mean(self.rocauc),
            'stddev_roc_auc': np.std(self.rocauc),
            'avg_pr_auc': np.mean(self.prauc),
            'stddev_pr_auc': np.std(self.prauc),
            'avg_roc_auc_0.1': np.mean(self.rocauc_0_1),
            'stddev_roc_auc_0.1': np.std(self.rocauc_0_1),
            'avg_elapsed_time': np.mean(self.elapsed_times),
            'stddev_elapsed_time': np.std(self.elapsed_times),
        }

    def before_fit(self):
        self.start_time = time.time()

    def after_fit(self):
        finish_time = time.time()
        self.elapsed_times.append(finish_time - self.start_time)

    def cross_validation(self, model_producer, interaction_matrix: np.ndarray,
                        n_drug, n_target, n_fold, validation_type):
        rand = self.random_state
        self.prepare()
        divs = cross_validation_divisions(rand, n_drug, n_target, n_fold, validation_type)

        for fold, division in enumerate(divs):
            division = division.T
            _im = interaction_matrix.copy()
            _im[division[0], division[1]] = 0

            model = model_producer()
            self.before_fit()
            model.fit(_im)
            self.after_fit()

            predicted = model.predict()
            predicted_tests = predicted[division[0], division[1]]
            true_tests = interaction_matrix[division[0], division[1]]

            sidx = np.argsort(-predicted_tests)
            predicted_tests = predicted_tests[sidx]
            division[0] = division[0, sidx]
            division[1] = division[1, sidx]
            true_tests = true_tests[sidx]

            self.after_test(fold, true_tests, predicted_tests)

        return self.after_all()
