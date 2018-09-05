import chainer
import chainer.functions as F
import chainer.links as L


class DropConvolution(chainer.Chain):
    def __init__(self, drop_rate, in_channels, out_channels, ksize, stride=1, pad=0, deconv=False):
        conv_link = L.Deconvolution2D if deconv else L.Convolution2D
        super().__init__(
            conv=conv_link(in_channels, out_channels, ksize, stride, pad),
        )
        self._drop_rate = drop_rate

    def __call__(self, x, test=False):
        return F.dropout(self.conv(x), self._drop_rate, not test)


class DL(chainer.Chain):
    def __init__(self, drop_rate, in_dim, out_dim):
        super().__init__(
            linear=L.Linear(in_dim, out_dim),
        )
        self._drop_rate = drop_rate

    def __call__(self, x, test=False):
        return F.dropout(self.linear(x), self._drop_rate, not test)


class AutoEncoder(chainer.Chain):
    def __init__(self, fp_dim, n_latent=128, n_layer=4):
        super().__init__()
        drop_rate = 0.5
        self.n_layer = n_layer
        output_dims = [fp_dim]
        self.layers = []
        for l in range(1, n_layer + 1):
            name = 'layer%d' % l
            output_dim = fp_dim if l == n_layer else fp_dim//2
            layer = DL(drop_rate, output_dims[-1], output_dim) if l != n_layer else DL(0, output_dims[-1], output_dim)
            self.layers.append(layer)
            self.add_link(name, layer)
            output_dims.append(output_dim)

        self.add_link('encode', L.Linear(fp_dim//2, n_latent))

    def __call__(self, x, test=False):
        outputs = [x]
        for l, layer in enumerate(self.layers, start=1):
            act_func = F.sigmoid if l == self.n_layer else F.relu
            outputs.append(act_func(layer(outputs[-1], test)))
        return self.encode(outputs[len(outputs)//2]), outputs[-1]


    def w_loss(self):
        xp = chainer.cuda.get_array_module(self.layers[0].linear.W.data)
        loss = chainer.Variable(xp.zeros(()).astype('f'))
        for p in self.params():
            loss += F.sum(F.square(p))
        return loss


class WangAutoEncoder(chainer.Chain):
    def __init__(self, fp_dim, n_latent, n_layer=4):
        super().__init__()
        drop_rate = 0.5
        output_dims = [fp_dim]
        self.layers = []
        for l in range(1, n_layer + 1):
            name = 'layer%d' % l
            output_dim = n_latent if l == n_layer//2 else (fp_dim if l == n_layer else 200)
            layer = DL(drop_rate, output_dims[-1], output_dim) if l != n_layer else DL(0, output_dims[-1], output_dim)
            self.layers.append(layer)
            self.add_link(name, layer)
            output_dims.append(output_dim)

    def __call__(self, x, test=False):
        outputs = [x]
        for layer in self.layers:
            outputs.append(F.sigmoid(layer(outputs[-1], test)))
        return outputs[len(outputs)//2], outputs[-1]

    def w_loss(self):
        xp = chainer.cuda.get_array_module(self.layers[0].linear.W.data)
        loss = chainer.Variable(xp.zeros(()).astype('f'))
        for p in self.params():
            loss += F.sum(F.square(p))
        return loss