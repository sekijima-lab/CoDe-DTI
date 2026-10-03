"""CPU autoencoders preserving the original architecture and NumPy RNG."""
import math
import numpy as np
import torch
from torch import nn


class DL(nn.Module):
    def __init__(self, drop_rate, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.from_numpy(np.random.normal(0, 1 / math.sqrt(in_dim), (out_dim, in_dim)).astype('f')))
        self.b = nn.Parameter(torch.zeros(out_dim))
        self._drop_rate = drop_rate

    def forward(self, x, test=False):
        # Separate bias addition, as in Chainer's original linear function.
        y = torch.matmul(x, self.W.T) + self.b
        # Even ratio zero consumes RNG in Chainer 1.24.
        if not test:
            mask = (np.random.rand(*y.shape) >= self._drop_rate).astype('f') / (1 - self._drop_rate)
            y = y * torch.from_numpy(mask)
        return y


class AutoEncoder(nn.Module):
    def __init__(self, fp_dim, n_latent=128, n_layer=4, _wang=False):
        super().__init__()
        if n_layer < 2 or n_layer % 2:
            raise ValueError('n_layer must be positive and even (at least 2)')
        self.n_layer = n_layer
        self._wang = _wang
        self.layers = nn.ModuleList()
        dim = fp_dim
        for l in range(1, n_layer + 1):
            out_dim = (n_latent if l == n_layer // 2 else (fp_dim if l == n_layer else 200)) if _wang else (fp_dim if l == n_layer else fp_dim // 2)
            self.layers.append(DL(0 if l == n_layer else .5, dim, out_dim))
            dim = out_dim
        if not _wang:
            self.encode = DL(0, fp_dim // 2, n_latent)

    def forward(self, x, test=False):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.device.type != 'cpu' or x.dtype != torch.float32:
            raise ValueError('Only CPU float32 is validated')
        outputs = [x]
        for i, layer in enumerate(self.layers, 1):
            y = layer(outputs[-1], test)
            # Chainer 1.24 CPU sigmoid uses tanh, rather than exp.
            outputs.append(torch.tanh(y * .5) * .5 + .5 if self._wang or i == self.n_layer else torch.relu(y))
        code = outputs[len(outputs) // 2]
        return (code if self._wang else self.encode(code, test=True)), outputs[-1]

    def named_legacy_parameters(self):
        values = []
        for i, layer in enumerate(self.layers, 1):
            values.extend([(f'/layer{i}/linear/W', layer.W), (f'/layer{i}/linear/b', layer.b)])
        if not self._wang:
            values.extend([('/encode/W', self.encode.W), ('/encode/b', self.encode.b)])
        return sorted(values)

    def w_loss(self):
        loss = torch.zeros((), dtype=torch.float32)
        for layer in list(self.layers) + ([] if self._wang else [self.encode]):
            loss = loss + torch.sum(layer.W.square())
            loss = loss + torch.sum(layer.b.square())
        return loss


class WangAutoEncoder(AutoEncoder):
    def __init__(self, fp_dim, n_latent, n_layer=4):
        super().__init__(fp_dim, n_latent, n_layer, _wang=True)
