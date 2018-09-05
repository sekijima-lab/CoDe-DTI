from collections import defaultdict
from typing import Set, List

import chainer
import numpy as np


def sample(lst: list, sample_size: int):
    return np.random.permutation(lst)[:sample_size].tolist()


def precision_recall_m(scores: np.ndarray, positives: Set[int], train_set: Set[int], ms: List[int]):
    sorted_items = [item for item in (-scores).argsort().tolist() if item not in train_set]
    true_positives = [0 for _ in range(max(ms) + 1)]
    for m in range(max(ms)):
        true_positives[m + 1] = true_positives[m] + (1 if sorted_items[m] in positives else 0)
    precisions = [true_positives[m] / m for m in ms]
    recalls = [true_positives[m] / len(positives) for m in ms] if len(positives) > 0 else None
    return precisions, recalls


def evaluate(n_users, learn_func, logs, sample_size):
    # sampled = [(u,) + tuple(v) for u, vs in logs.items() for v in sample(vs, 10)]
    sampled = []
    train_sets = defaultdict(set)
    for u, vs in logs.items():
        for v in sample(vs, sample_size):
            sampled.append((u,) + tuple(v))
            train_sets[u].add(v[0])
    c = learn_func(sampled)

    # TODO: calc metrics
    pms = []
    rms = []
    for u in range(n_users):
        xp = chainer.cuda.get_array_module(c.u)
        scores = chainer.cuda.to_cpu(xp.matmul(c.u[u], c.v.T))
        positives = set(p[0] for p in logs[u]) - train_sets[u]
        precision, recall = precision_recall_m(scores, positives, train_sets[u], [10, 20, 30, 40, 50, 100])
        pms.append(precision)
        if recall is not None:
            rms.append(recall)
    print('precision:', np.asarray(pms).mean(axis=0))
    print('variance of precision:', np.asarray(pms).var(axis=0))
    print('recall:', np.asarray(rms).mean(axis=0))
    print('variance of recall:', np.asarray(rms).var(axis=0))
