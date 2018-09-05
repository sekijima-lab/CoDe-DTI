from typing import Union

import chainer
import chainer.functions as F
from chainer import training, reporter
from chainer.dataset import dataset_mixin
import numpy as np


class FingerprintDataset(dataset_mixin.DatasetMixin):
    def __init__(self, path):
        self._fingerprints = []
        with open(path) as fp:
            for line in fp:
                fp_str = line.strip().split(' ')[1]
                self._fingerprints.append(np.array(list(map(float, fp_str)), dtype=np.float32))

    def __len__(self):
        return len(self._fingerprints)

    def get_example(self, i):
        return i, self._fingerprints[i]


class CDLUpdater(training.StandardUpdater):
    def __init__(self, ctr, encoder, logs, iterator, optimizer, device, **cdl_parameters):
        super().__init__(iterator, optimizer, device=device)
        self._dataset = iterator.dataset
        self.logs = logs
        self.ctr = ctr
        self.encoder = encoder

        if cdl_parameters is None:
            cdl_parameters = {}

        self._lw = cdl_parameters.pop('lambda_w', 1.0)
        self._lv = cdl_parameters.pop('lambda_v', 1.0)
        self._ln = cdl_parameters.pop('lambda_n', 4.0)

    def update_theta(self):
        it = chainer.iterators.SerialIterator(self._dataset, batch_size=50, repeat=False, shuffle=False)
        for batch in it:
            batch = self.converter(batch, self.device)
            batch = (chainer.cuda.to_cpu(batch[0]), batch[1])
            new_code, _ = self.encoder(chainer.Variable(batch[1], volatile='on'), test=True)
            self.ctr.theta[batch[0]] = chainer.cuda.to_cpu(new_code.data)

    def update_core(self):
        batch = self.converter(self.get_iterator('main').next(), self.device)  # type: Union[cupy.ndarray, np.ndarray]
        batch_size = len(batch[1])
        batch = (chainer.cuda.to_cpu(batch[0]), batch[1])
        opt = self.get_optimizer('main')  # type: chainer.optimizer.Optimizer

        if self.is_new_epoch:
            # Update theta, U, V
            self.update_theta()
            self.ctr.als(self.logs)
            err = 0
            # err = self.ctr.error(self.logs)
            reporter.report({'ctr_error': err}, opt.target)

        # Update theta
        code, recons = self.encoder(batch[1])
        recons_loss = F.sum(F.square(batch[1] - recons))/batch_size
        if self.device is None:
            code_loss = F.sum(F.square(code - self.ctr.v[batch[0]]))/batch_size
        else:
            code_loss = F.sum(F.square(code - chainer.cuda.to_gpu(self.ctr.v[batch[0]], device=self.device)))/batch_size
        regularization = self.encoder.w_loss()
        if self._lv != 0:
            loss = self._lv * code_loss + self._ln * recons_loss + self._lw * regularization  # type: chainer.Variable
        else:
            loss = self._ln * recons_loss + self._lw * regularization  # type: chainer.Variable
        reporter.report({
            'loss': loss,
            'recons_loss': recons_loss,
            'code_loss': code_loss,
            'regularization_loss': regularization,
        }, opt.target)

        opt.target.cleargrads()
        loss.backward()
        opt.update()
