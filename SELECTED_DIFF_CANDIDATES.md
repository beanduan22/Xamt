# Selected Xamt DIFF Candidates

Snapshot: `2026-05-25 10:43:06 AEST`
Source run: `/tmp/xamt_wide_arity_nonfinite_20260525_091040_seed20260539_auto`

These are the best candidates to minimize first after adapter cleanup. P0 means start here.

| Priority | Key | Suspect | Why It Is Worth Minimizing |
| --- | --- | --- | --- |
| P0 | `signbit/generic/1` | MindSpore | `signbit(-0.0)` differs from JAX/Keras; this is a tiny scalar-style IEEE edge case and should minimize cleanly. |
| P0 | `diagflat/generic/2` | MindSpore | Off-diagonal values become `NaN` where JAX/Keras/MXNet/NumPy keep zeros. Strong masked-fill/nonfinite propagation signal. |
| P0 | `tril/generic/2` | MindSpore | Masked upper-triangle entries inherit `NaN`; other libs zero the masked region. Small matrix reproducer likely. |
| P0 | `maximum/generic/2` and `minimum/generic/2` | MindSpore | MindSpore suppresses/chooses around `NaN` where Chainer/Keras/MXNet preserve `NaN`. Good binary-op edge candidate. |
| P1 | `clip/generic/3` | MindSpore | `clip(NaN, -1, 2)` returns `-1` while Chainer/JAX/Keras preserve `NaN`. Very small reproducer. |
| P1 | `nancumsum/generic/3` | MindSpore | Keras and MindSpore diverge on `NaN`/`Inf` cumulative behavior; likely reducible to a 1-D vector. |
| P1 | `angle/generic/2` | Paddle/TensorFlow vs NumPy/JAX | Difference appears tied to signed zero / negative real inputs. Needs doc check, but reproducer should be tiny. |
| P1 | `count_nonzero/reduction/3` | JAX vs NumPy/MXNet/MindSpore | Count differs on edge-valued input; likely a one-array minimal case. |
| P2 | `ceil/generic/1`, `ceil/generic/2` | JAX/Keras/TensorFlow vs MindSpore/MXNet/Chainer | Subnormal flushing vs mathematical ceil; may be backend policy rather than bug. |
| P2 | `log2/generic/*`, `log10/generic/*` | Multiple libs | Subnormal/underflow behavior differs; useful but less clean because backend flush-to-zero policy may explain it. |
| P2 | `logm/linalg/2` | SciPy vs TensorFlow | Complex branch sign differs. Interesting but branch convention may be documented. |
| P2 | `lstsq/linalg/5` | Paddle vs TensorFlow | Large numeric delta on likely ill-conditioned input; needs condition-number check before treating as bug. |

Recommended first minimization order:

1. `signbit/generic/1`
2. `diagflat/generic/2`
3. `tril/generic/2`
4. `clip/generic/3`
5. `maximum/generic/2` / `minimum/generic/2`
6. `nancumsum/generic/3`

Deprioritize for now: `split/shape/4` is adapter normalization, and NaN ordering cases like `argmax/argmin/argsort` need documentation checks before filing.
