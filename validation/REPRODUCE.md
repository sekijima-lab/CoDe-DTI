# Repeat the CPU comparisons

Use the root README for the maintained isolated installation. The old environment
is a separate Python 3.10.19 virtual environment; install exact `old-pins.txt`.
Chainer's undeclared build dependency requires setuptools 69.5.1 and wheel before
installing Chainer with `--no-build-isolation`. Never downgrade a research environment.
Archive master commit `438a2be38c62bc9a6ff4db7e16416fb6d039d2a8` to an independent
`code-dti-reference` directory. Commands below run from the parent of the updated
`CoDe-DTI` checkout; substitute isolated interpreter paths as needed.

```sh
mkdir -p code-dti-runtime-validation
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR="$PWD/code-dti-runtime-validation/mpl"
code-dti-old/bin/python CoDe-DTI/validation/benchmark.py code-dti-reference code-dti-runtime-validation/old.npz
code-dti-modern/bin/python CoDe-DTI/validation/benchmark.py CoDe-DTI code-dti-runtime-validation/new.npz --modern
code-dti-modern/bin/python CoDe-DTI/validation/compare.py code-dti-runtime-validation/old.npz code-dti-runtime-validation/new.npz
code-dti-old/bin/python CoDe-DTI/validation/train_benchmark.py code-dti-reference code-dti-runtime-validation/old-train.npz
code-dti-modern/bin/python CoDe-DTI/validation/train_benchmark.py CoDe-DTI code-dti-runtime-validation/new-train.npz --modern
code-dti-modern/bin/python CoDe-DTI/validation/compare.py code-dti-runtime-validation/old-train.npz code-dti-runtime-validation/new-train.npz --training
```

Run old and new training sequentially: they write the same fixture directory.
Alternatively, compare current outputs to compact committed references with
`reference.npz` and `training-reference.npz` as the first compare argument.
The compact comparisons check samples for large parameters/gradients; full
workspace arrays were used for the reported maxima. Forward arrays remain full.
`fixtures/indices.json` records the 120-drug × 12-target subset of bundled data.

`cv_benchmark.py` runs the actual two-fold library evaluation (use --modern for
the updated environment); the README `albrecht` command exercises the installed
console entry point. `nfp_benchmark.py` repeats the experimental fingerprint
forward comparison. Each run creates its own output files/directories.
For random search, provide the normal configuration.json schema with method
random and max_iter=1 for a smoke run, pointing to the committed fixtures.
`giselle CONFIG.json --output-dir OUTPUT` runs a trial with subprocess/spawn.
Do not run the historical default 200-trial sweep just to verify installation.
