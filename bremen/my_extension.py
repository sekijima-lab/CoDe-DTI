"""Image evaluation callback for optional autoencoder experiments."""
import os
import numpy as np
import torch
from PIL import Image

def _save_img(img, path):
    Image.fromarray(img.transpose(1, 2, 0)).save(path)

def eval_image(model, dataset, n_sample=100, batch_size=50, device=None, dirname_format='eval_{.updater.epoch}'):
    if device is not None:
        raise ValueError('Only CPU is validated')
    samples = [dataset[i] for i in np.random.randint(0, len(dataset), n_sample)]
    first_epoch = True
    def callback(trainer):
        nonlocal first_epoch
        dirname = os.path.join(trainer.out, dirname_format.format(trainer))
        os.makedirs(dirname, exist_ok=True)
        for start in range(0, len(samples), batch_size):
            features = np.asarray([sample[1] for sample in samples[start:start+batch_size]], dtype='f')
            if first_epoch:
                input_dir = os.path.join(trainer.out, 'eval_input');os.makedirs(input_dir, exist_ok=True)
                for i, img in enumerate(((features + 1) * 127.5).astype('uint8')):
                    _save_img(img, os.path.join(input_dir, f'{start+i}.png'))
            with torch.no_grad():
                generated = model(torch.from_numpy(features), test=True)[1].numpy()
            for i, img in enumerate(((generated + 1) * 127.5).astype('uint8')):
                _save_img(img, os.path.join(dirname, f'{start+i}.png'))
        first_epoch = False
    return callback
