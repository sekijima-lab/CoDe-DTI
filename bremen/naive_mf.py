import numpy as np
import chainer.cuda
from typing import Dict, Iterable, Tuple, Union, List
from collections import defaultdict
from scipy.sparse import coo_matrix
from . import opt
if chainer.cuda.available:
    import cupy


class NaiveMF:
    def __init__(self, n_latent, n_user, n_item, **params):
        xp = np
        self._U = xp.random.randn(n_user, n_latent).astype('f')
        self._V = xp.random.randn(n_item, n_latent).astype('f')
        self._n_latent = n_latent

        self._lu = params.pop('lambda_u', 1.0)
        self._lv = params.pop('lambda_v', 1.0)
        self._alpha = params.pop('alpha', 40.0)

        self._lambda_u_eye = xp.identity(n_latent, xp.float32)
        xp.multiply(self._lu, self._lambda_u_eye, self._lambda_u_eye)
        self._lambda_v_eye = xp.identity(n_latent, xp.float32)
        xp.multiply(self._lv, self._lambda_v_eye, self._lambda_v_eye)

    """ Alternating Least Square
    :param logs: iterable collection of tuple[user_id, item_id, rating]
    """
    def als(self, logs: Iterable[Tuple[int, int, int]]):
        user_log = defaultdict(list)
        item_log = defaultdict(list)
        for u, i, r in logs:
            user_log[u].append((i, r))
            item_log[i].append((u, r))

        # User update
        self._update(self._U, self._V, self._lambda_u_eye, user_log)

        # Item update
        self._update(self._V, self._U, self._lambda_v_eye, item_log)

    @property
    def v(self):
        return self._V

    @property
    def u(self):
        return self._U

    def save_npz(self, filename: str):
        np.savez_compressed(filename,
                            v=chainer.cuda.to_cpu(self._V),
                            u=chainer.cuda.to_cpu(self._U))

    def load_npz(self, filename: str):
        data = np.load(filename)
        self._V = data['v']
        self._U = data['u']

    def error(self, logs: Iterable[Tuple[int, int, int]]):
        user_log = defaultdict(list)
        for u, i, r in logs:
            user_log[u].append((i, r))

        err = []
        for u_idx in range(self._U.shape[0]):
            u = self._U[u_idx]
            r_ = np.matmul(u, self._V.T)
            truth = np.zeros_like(r_)
            confidence = np.ones_like(r_)
            for i, r in user_log.get(u_idx, []):
                truth[i] = 1 if r > 0 else 0
                confidence[i] += self._alpha * r
            err.append(np.mean(confidence*np.square(r_ - truth)))
        return np.mean(err)

    def _update(self, x, y, lambda_eye, logs: Dict[int, List[Tuple[int, int]]]):
        xp = chainer.cuda.get_array_module(x)  # type: Union[np, cupy]
        YtY = xp.matmul(y.T, y)
        YtY_tmp = xp.empty_like(YtY, np.float32)
        # YtY_delta = xp.empty_like(YtY, np.float32)
        YCP = xp.empty(self._n_latent, np.float32)
        YCP_delta = xp.empty_like(YCP, np.float32)

        for x_idx in range(x.shape[0]):
            log = logs[x_idx]
            xp.copyto(YtY_tmp, YtY)
            YCP.fill(0)

            for y_idx, r in log:
                if r == 0:
                    continue
                yj = y[y_idx]

                # calc transpose(Y)CY
                yj.shape = (1, yj.size)
                YtY_delta = xp.matmul(yj.T, yj)
                xp.multiply(self._alpha * r, YtY_delta, YtY_delta)
                xp.add(YtY_tmp, YtY_delta, YtY_tmp)

                # calc transpose(Y)CP
                yj.shape = (yj.size,)
                xp.multiply(1.0 + self._alpha * r, yj, YCP_delta)
                xp.add(YCP, YCP_delta, YCP)

            xp.add(YtY_tmp, lambda_eye, YtY_tmp)
            opt.solve(YtY_tmp, YCP)
            x[x_idx] = YCP

class NaiveMFDriver:
    def __init__(self, n_latent, n_drug, n_target, epoch, **parameters):
        self.model = NaiveMF(n_latent, n_target, n_drug, **parameters)
        self.stop_epoch = epoch
        self.parameters = parameters

    def fit(self, intmat):
        coo = coo_matrix(intmat)
        logs = [(target, drug, data) for target, drug, data in zip(coo.col, coo.row, coo.data)]
        for epoch in range(self.stop_epoch):
            self.model.als(logs)

    def predict(self):
        u = self.model.u
        v = self.model.v
        return np.matmul(u, v.T).T