# CoDe-DTI: maintained CPU runtime

The paper reproduction version remains on **master**. This branch,
`security/modern-runtime`, is maintained separately and is **not intended to
merge into master**. It updates Python and dependencies and replaces Chainer
1.24 with PyTorch while retaining the collaborative deep learning algorithm.

Use an isolated environment; existing research environments need not change:

```sh
uv venv --python 3.12.15 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-build-isolation .
uv pip check --python .venv/bin/python
```

Python 3.12.15; PyTorch 2.14.1; NumPy 2.5.3; SciPy 1.18.1;
scikit-learn 1.9.1; RDKit 2026.3.6; Matplotlib 3.11.2; Pillow 12.3.0.
`requirements-lock.txt` pins all 28 external distributions. The local `bremen`
package is version 0.2.0. Runtime scope is CPU float32; GPU calls fail explicitly.

```sh
.venv/bin/albrecht --seed 19 --encoder wang --n-fold 2 --epoch 1 \
  --n-latent 8 --alpha 4 --lambda-n 4 --lambda-w .5 --lambda-v .5 --lambda-u 1 \
  --output-dir validation/cli-output \
  validation/fixtures/intmat.npy validation/fixtures/fingerprints.txt
.venv/bin/python -m unittest discover -s tests -v
```

The NumPy weight initializer, tanh-based historical sigmoid, Dropout masks and
random consumption order, epoch wrapping, SGD and alternating least squares
retain the original CPU semantics. `albrecht` records the seed and versions in
`runtime.json` and uses one PyTorch CPU thread. Library callers should configure
one CPU thread for the validated numerical scope. `fit` retains per-step metrics
in `training_log`; the former Chainer Trainer/extension interface is replaced.
Numeric NPZ loading always uses `allow_pickle=False`.

`giselle` supports the original random parameter search. The source repository
never supplied `parameter_search/bayes.py`; GP search raises an explicit error.
Its queue is passed to the child process so macOS spawn works correctly.
The obsolete histogram `normed` keyword is replaced with `density`.
The historical `avg_roc_auc_0.1` field still computes full ROC AUC, as in master;
it should not be interpreted as partial AUC. This migration does not change it.

Validation covers three seeds, both autoencoder architectures, forward/backward
and SGD, ALS, two-epoch collaborative training on 120 real drugs × 12 targets,
and actual two-fold CLI cross-validation. All frozen numerical criteria pass.
See [validation/REPORT.md](validation/REPORT.md) and
[validation/REPRODUCE.md](validation/REPRODUCE.md). This is a reconstructed CPU
comparison, not a guarantee of the original paper's full training trajectory.
The experimental NFP helper has forward comparison and a finite-gradient test;
its Adam training, image callbacks, CUDA, complete parameter sweeps, full dataset
training and GitHub CI are not validated. Use master for historical GPU and
Chainer APIs, including the old convolution helper.
