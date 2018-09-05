import chainer
from chainer import training
import numpy as np
import os
from PIL import Image
from typing import Iterable


def _save_img(img: np.ndarray, path):
    img = img.transpose(1, 2, 0)
    image = Image.fromarray(img)
    image.save(path)


def eval_image(model, dataset, n_sample=100, batch_size=50, device=None, dirname_format='eval_{.updater.epoch}'):
    idx = np.random.randint(0, len(dataset), n_sample)  # type: Iterable[int]
    sample_images = [dataset[i] for i in idx]
    first_epoch = True

    @training.make_extension(trigger=(1, 'epoch'), default_name='eval_image')
    def func(trainer: training.Trainer):
        nonlocal first_epoch
        dirname = os.path.join(trainer.out, dirname_format.format(trainer))
        os.makedirs(dirname, exist_ok=True)

        for i in range(0, len(sample_images), batch_size):
            batch = sample_images[i:i + batch_size]
            batch = chainer.dataset.convert.concat_examples(batch, device=device)

            if first_epoch:
                input_dir = os.path.join(trainer.out, 'eval_input')
                os.makedirs(input_dir, exist_ok=True)
                input_ = ((chainer.cuda.to_cpu(batch[1]) + 1.0) * 127.5).astype('uint8')
                for j, img in enumerate(input_):
                    _save_img(img, os.path.join(input_dir, '{}.png'.format(i + j)))

            generated = chainer.cuda.to_cpu(model(chainer.Variable(batch[1], volatile='on'), test=True)[1].data)
            generated = ((generated + 1.0) * 127.5).astype('uint8')
            for j, img in enumerate(generated):
                _save_img(img, os.path.join(dirname, '{}.png'.format(i + j)))
        first_epoch = False

    return func
