"""Collaborative deep learning with the original CPU iteration order."""
import numpy as np
import torch


class FingerprintDataset:
    def __init__(self, path):
        with open(path) as fp:
            self._fingerprints = [np.asarray(list(map(float, line.strip().split(' ')[1])), dtype='f') for line in fp]
        if not self._fingerprints:
            raise ValueError('Empty fingerprint dataset')
    def __len__(self):
        return len(self._fingerprints)
    def __getitem__(self, i):
        return i, self._fingerprints[i]
    get_example = __getitem__


class SerialIterator:
    """Preserve original NumPy shuffle and wrap semantics (including short data)."""
    def __init__(self, dataset, batch_size=100):
        self.dataset = dataset
        self.batch_size = batch_size
        self.order = np.random.permutation(len(dataset))
        self.position = self.epoch = 0
    def next(self):
        end = self.position + self.batch_size
        batch = [self.dataset[i] for i in self.order[self.position:end]]
        new_epoch = end >= len(self.dataset)
        if new_epoch:
            rest = end - len(self.dataset)
            np.random.shuffle(self.order)
            batch.extend(self.dataset[i] for i in self.order[:rest])
            self.position = rest
            self.epoch += 1
        else:
            self.position = end
        return batch, new_epoch


class CDLUpdater:
    def __init__(self, ctr, encoder, logs, iterator, optimizer, device=None, **params):
        if device is not None:
            raise ValueError('Only CPU is validated')
        self.ctr, self.encoder, self.logs = ctr, encoder, logs
        self.iterator, self.optimizer = iterator, optimizer
        self._lw, self._lv, self._ln = params.get('lambda_w', 1.), params.get('lambda_v', 1.), params.get('lambda_n', 4.)
    def update_theta(self):
        with torch.no_grad():
            for start in range(0, len(self.iterator.dataset), 50):
                batch = [self.iterator.dataset[i] for i in range(start, min(start+50, len(self.iterator.dataset)))]
                ids, features = zip(*batch)
                code, _ = self.encoder(np.asarray(features, dtype='f'), test=True)
                self.ctr.theta[list(ids)] = code.numpy()
    def update_core(self):
        batch, new_epoch = self.iterator.next()
        ids, features = zip(*batch)
        x = torch.from_numpy(np.asarray(features, dtype='f'))
        if new_epoch:
            self.update_theta()
            self.ctr.als(self.logs)
        code, recons = self.encoder(x)
        recons_loss = (x - recons).square().sum() / len(batch)
        code_loss = (code - torch.from_numpy(self.ctr.v[list(ids)])).square().sum() / len(batch)
        regularization = self.encoder.w_loss()
        loss = self._ln * recons_loss + self._lw * regularization
        if self._lv != 0:
            # Preserve the original addition order.
            loss = self._lv * code_loss + self._ln * recons_loss + self._lw * regularization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {name: float(value.detach()) for name, value in dict(loss=loss, recons_loss=recons_loss,code_loss=code_loss,regularization_loss=regularization).items()}
