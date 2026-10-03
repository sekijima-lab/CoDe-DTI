# CoDe-DTI CPU dependency migration validation

## Policy and reference

Master remains at `438a2be38c62bc9a6ff4db7e16416fb6d039d2a8`, the original
IEEE BIBM implementation. The updated runtime is maintained only on
`security/modern-runtime`; do not merge it into master.
No pretrained autoencoder/checkpoint or complete original dependency export was
provided. The only historical exact runtime pin is `chainer==1.24` in setup.py.
The independent reference uses Python 3.10.19, Chainer 1.24.0, NumPy 1.23.5,
SciPy 1.10.1, scikit-learn 1.2.2, Matplotlib 3.7.5 and RDKit 2022.9.5.
Other reference packages are recorded in `old-pins.txt`. These are reconstructed
CPU dependencies, not the author's exact 2018 environment. Source and bundled
input SHA256 hashes are recorded in `provenance.json`.

Updated: Python 3.12.15, 28 pinned external PyPI distributions and local bremen
0.2.0. A fresh isolated installation passed all 29-distribution dependency checks.
Existing research environments were not modified.

## Security scope

Official OSV name/version queries on 2026-10-03 matched two packages in the
reconstructed reference (scikit-learn and setuptools), and none of the 28 new
pins. `osv-snapshot.json` retains the exact queries/results. The original unpinned
versions cannot be reconstructed from repository metadata. This is not a full
native/Python/OS/application audit or a guarantee against future advisories.

The scikit-learn advisory
[CVE-2024-5206](https://github.com/advisories/GHSA-jw8x-6495-233v) concerns
TfidfVectorizer, which this implementation does not use. Updating this package
and the vulnerable reference build tooling is preventative; no exploitable
vectorizer use was identified. Chainer is removed from maintained dependencies.
Numeric input and matrix state loading use `allow_pickle=False`; object arrays
are rejected by regression tests. There is no model-pickle conversion or load.

## Criteria and independent comparisons

`criteria.json` was frozen before comparing old and new arrays. It was not
relaxed. CPU float32, one CPU thread and seeds 0/19/73 were used. Same input,
initialization and NumPy-generated masks isolate arithmetic from RNG differences.
Full-array comparisons support the results below. Committed baselines retain
all forward inputs/outputs and sampled large parameter/gradient arrays (up to
4096 elements each, explicit indices); full arrays are retained in the workspace.

For both bremen and wang architectures, 1024-bit real bundled fingerprints,
eight examples, latent dimension 8 and four layers:

- Initial parameters, input fingerprints and all three CV split modes: exact.
- Forward outputs, maximum absolute difference: 3.3155083656311035e-7 (limit 1e-5).
- Loss: 6.103515625e-5 (limit 1e-4).
- Full gradients: 8.940696716308594e-7 (limit 1e-4).
- One complete SGD step: 1.4901161193847656e-8 (limit 1e-4).
- Independent ALS step: U/V exact (limit 1e-4).

Two-epoch alternating collaborative training uses 120 real drugs and 12 targets,
394 positive interactions, batch size 100, both architectures and three seeds.
The subset selects the 12 targets with most positive interactions, then 120 drugs
with most positives for these targets, stable ordering. Exact source indices
and input fixtures are committed. Maximum differences:

- Theta codes: 3.129243850708008e-7.
- U: 1.946091651916504e-5; V: 4.0411949157714844e-5 (limit 1e-4).
- Final prediction matrix: 2.130866050720215e-5 (limit 1e-4).
- Complete autoencoder weights: 5.960464477539063e-8 (limit 1e-4).

All frozen criteria pass. Actual installed `albrecht` CLI: two-fold CV on this
real subset, wang architecture, seed 19, one epoch. ROC AUC/PR AUC and their
standard deviations match exactly (`cv-comparison.json`), excluding elapsed time.
A complementary synthetic-interaction test with real fingerprints also passes
three-seed training criteria; its CV metrics differ slightly, with maximum
PR AUC difference 5.655982018345e-5. Ranking/metrics are not universally invariant.

## RNG cause and runtime checks

A first multi-step comparison differed substantially despite a passing single
step. Chainer 1.24 draws Dropout masks even at ratio zero. Skipping those draws
shifted every later mask. Restoring the zero-ratio RNG consumption removed this
cause; failed measurements are retained in `training-rng-failed.json`.
The original sigmoid uses `tanh(x/2)/2+1/2`, and linear bias addition is separate.
NumPy normal initialization, mask RNG, parameter-regularization order, wrapping
and ALS-before-SGD at an epoch boundary are retained. No global monkeypatch or
old library binary is used.

Fresh isolated environment: six regression tests pass (reference forward and
gradients for both architectures, zero-ratio RNG, iterator wrapping, ALS/state
roundtrip/object rejection, neural fingerprint gradients, unsupported CPU inputs).
The installed `giselle` random-search CLI completed one real-data trial with
macOS spawn. GP exploration was already unavailable because its module is absent.
The experimental NFP forward comparison spans three molecules × three seeds,
maximum difference 7.152557373046875e-7. NFP Adam equivalence is not established.

## Limits

Validated on macOS ARM64 CPU float32 only. Full 5877-drug × 3348-target dataset
training, paper hyperparameter sweeps, Linux/CUDA, higher-order derivatives,
experimental NFP Adam learning, image callbacks and GitHub CI are not validated.
DropConvolution/Chainer Trainer APIs and the CUDA 8 solver remain on master;
the maintained runtime exposes CPU learning and explicitly rejects GPU solves.
No claim of paper-wide or universal bitwise reproducibility is made.

Primary arithmetic references: original repo source and Chainer 1.24
[sigmoid](https://docs.chainer.org/en/v1.24.0/_modules/chainer/functions/activation/sigmoid.html),
[Dropout](https://docs.chainer.org/en/v1.24.0/_modules/chainer/functions/noise/dropout.html),
[SerialIterator](https://docs.chainer.org/en/v1.24.0/_modules/chainer/iterators/serial_iterator.html).
