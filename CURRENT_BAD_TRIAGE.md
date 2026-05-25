# Current Xamt Bad-Case Triage

Snapshot: `2026-05-25 09:55:19 AEST`
Run dir: `/tmp/xamt_wide_arity_nonfinite_20260525_091040_seed20260539_auto`

## Current Progress

| State | Groups |
| --- | ---: |
| PASS | 443 |
| DIFF | 37 |
| ERROR | 54 |
| SKIP | 87 |
| TOTAL_WRITTEN | 621 |

Shard lines: shard0=155, shard1=157, shard2=153, shard3=156

## Triage Summary

| Class | Count |
| --- | ---: |
| DIFF: NaN/order semantics | 4 |
| DIFF: adapter normalization | 1 |
| DIFF: nonfinite propagation | 16 |
| DIFF: numeric/branch semantics | 2 |
| DIFF: special/edge semantics | 5 |
| DIFF: value semantics | 9 |
| adapter/input-plan | 35 |
| backend/env limitation | 14 |
| needs manual confirmation | 5 |

## Per-Case Pass

| # | State | Key | Triage | Shard | Seed | Libs | Note |
| ---: | --- | --- | --- | --- | ---: | --- | --- |
| 1 | ERROR | `fold/nn/6` | adapter/input-plan | shard0 | 129260539 | mindspore,torch | mindspore: mindspore.mint.nn.functional.fold: RuntimeError: TypeError: fold_ext() missing 2 required positional arguments: 'output_size' and 'kernel_size'; tor... |
| 2 | ERROR | `pad/nn/4` | adapter/input-plan | shard0 | 141260539 | mindspore,torch | mindspore: mindspore.mint.nn.functional.pad: RuntimeError: TypeError: Failed calling ConstantPadND with "ConstantPadND()(input=Tensor, padding=Tuple<Tuple<int,... |
| 3 | ERROR | `upsample/nn/5` | needs manual confirmation | shard0 | 153260539 | mindspore,torch | mindspore: mindspore.mint.nn.functional.upsample: RuntimeError: ValueError: For 'interpolate', 'size' and 'scale_factor' cannot be both empty; torch: torch.nn.... |
| 4 | DIFF | `argmax/reduction/4` | DIFF: NaN/order semantics | shard0 | 161260540 | jax,numpy,tensorflow | jax=[2] [0, 1]; numpy=[2] [2, 1]; tensorflow=[2] [0, 1] |
| 5 | DIFF | `multiply/generic/3` | DIFF: nonfinite propagation | shard0 | 197260650 | mindspore,mxnet,tensorflow | mindspore=[2, 3] [[nan, nan, -inf], [0.18958938121795654, 0.2641873359680176...; mxnet=[2, 3] [[nan, nan, -inf], [0.18958938121795654, 0.2641873359680176...; tensorflow=[2, 3] [[n... |
| 6 | DIFF | `angle/generic/2` | DIFF: value semantics | shard0 | 237260541 | jax,numpy,paddle,tensorflow | jax=[2, 3] [[3.1415927410125732, 0.0, 0.0], [3.1415927410125732, 0.0, ...; numpy=[2, 3] [[3.1415927410125732, 0.0, 0.0], [3.1415927410125732, 0.0, ...; paddle=[2, 3] [[0.0, 0.0, 0... |
| 7 | DIFF | `digamma/generic/2` | DIFF: special/edge semantics | shard0 | 241260544 | paddle,tensorflow | paddle=[2, 3] [[inf, -inf, -inf], [nan, -0.5772159099578857, -1000000.562...; tensorflow=[2, 3] [[nan, nan, nan], [nan, -0.5772159099578857, -1000000.5625]] |
| 8 | ERROR | `log_softmax/nn/3` | adapter/input-plan | shard0 | 249260539 | jax,mindspore,tensorflow | mindspore: mindspore.mint.special.log_softmax: RuntimeError: TypeError: log_softmax_ext() got an unexpected keyword argument 'axis' |
| 9 | DIFF | `softmax/nn/3` | DIFF: nonfinite propagation | shard0 | 261260542 | jax,mindspore,tensorflow | jax=[2, 3] [[nan, nan, nan], [0.23795250058174133, 0.3336610198020935,...; mindspore=[2, 3] [[nan, nan, 0.0], [0.23795250058174133, 0.33366093039512634...; tensorflow=[2, 3] [[nan... |
| 10 | DIFF | `lstsq/linalg/5` | DIFF: numeric/branch semantics | shard0 | 273261222 | paddle,tensorflow | paddle=[2, 1] [[-1134.48486328125], [294.6731872558594]]; tensorflow=[2, 1] [[-1099.765869140625], [285.66302490234375]] |
| 11 | ERROR | `argmax/reduction/3` | adapter/input-plan | shard0 | 397260539 | jax,keras,mindspore,mxnet | mindspore: mindspore.mint.argmax: RuntimeError: TypeError: argmax_ext() got an unexpected keyword argument 'axis' |
| 12 | ERROR | `fold/generic/6` | adapter/input-plan | shard0 | 453260539 | keras,mindspore | keras: keras.ops.fold: TypeError: fold() missing 2 required positional arguments: 'output_size' and 'kernel_size'; mindspore: mindspore.ops.functional.fold: Ru... |
| 13 | DIFF | `isclose/generic/5` | DIFF: value semantics | shard0 | 473260558 | jax,keras,mindspore,mxnet,numpy | jax=[2, 3] [[False, True, True], [True, True, True]]; keras=[2, 3] [[False, False, False], [True, True, True]]; mindspore=[2, 3] [[False, True, True], [True, True, True]]; +2 libs |
| 14 | ERROR | `jvp/generic/4` | adapter/input-plan | shard0 | 481260539 | keras,mindspore | keras: keras.ops.jvp: TypeError: jvp() missing 2 required positional arguments: 'primals' and 'tangents'; mindspore: mindspore.jvp: RuntimeError: TypeError: jv... |
| 15 | ERROR | `logsumexp/reduction/3` | adapter/input-plan | shard0 | 501260539 | keras,mindspore | mindspore: mindspore.mint.logsumexp: RuntimeError: TypeError: logsumexp_ext() got an unexpected keyword argument 'axis' |
| 16 | DIFF | `multiply/generic/2` | DIFF: nonfinite propagation | shard0 | 509260618 | jax,keras,mindspore,mxnet | jax=[2, 3] [[nan, nan, nan], [-19.644245147705078, -0.0336472429335117...; keras=[2, 3] [[nan, nan, nan], [-19.644245147705078, -0.0336472429335117...; mindspore=[2, 3] [[nan, nan... |
| 17 | DIFF | `nonzero/generic/1` | DIFF: value semantics | shard0 | 517260541 | keras,mxnet,numpy | keras=[3, 2] [[1, 0], [1, 1], [1, 2]]; mxnet=[4, 2] [[0, 2], [1, 0], [1, 1], [1, 2]]; numpy=[4, 2] [[0, 2], [1, 0], [1, 1], [1, 2]] |
| 18 | DIFF | `rsqrt/generic/1` | DIFF: special/edge semantics | shard0 | 537260549 | chainer,keras,mindspore | chainer=[2, 3] [[-inf, inf, 2.671373844909537e+22], [nan, 1.0, 999.9999389...; keras=[2, 3] [[-inf, inf, inf], [nan, 1.0, 999.9999389648438]]; mindspore=[2, 3] [[-inf, inf, inf], ... |
| 19 | DIFF | `signbit/generic/1` | DIFF: value semantics | shard0 | 541260545 | jax,keras,mindspore | jax=[2, 3] [[True, False, False], [True, False, False]]; keras=[2, 3] [[True, False, False], [True, False, False]]; mindspore=[2, 3] [[False, False, False], [True, False, False]] |
| 20 | ERROR | `gamma/random/4` | adapter/input-plan | shard0 | 581260539 | jax,keras,mindspore | mindspore: mindspore.ops.gamma: RuntimeError: TypeError: gamma() missing 2 required positional arguments: 'alpha' and 'beta' |
| 21 | ERROR | `apply_over_axes/generic/3` | needs manual confirmation | shard0 | 589260539 | jax,mindspore,mxnet,numpy | mindspore: mindspore.numpy.apply_over_axes: RuntimeError: TypeError: Failed calling sum with "sum(axis=int, out=None)". The valid calling should be: "Tensor.su... |
| 22 | DIFF | `count_nonzero/reduction/3` | DIFF: value semantics | shard0 | 601260541 | jax,mindspore,mxnet,numpy | jax=[] 3; mindspore=[] 4; mxnet=[] 4; +1 libs |
| 23 | ERROR | `atleast_1d/generic/0` | backend/env limitation | shard1 | 34260539 | jax,mindspore,mxnet,numpy,torch | mindspore: mindspore.numpy.atleast_1d: RuntimeError: SystemError: <built-in method pyboost_reshape of PyCapsule object at 0x71b97df86310> returned a result wit... |
| 24 | ERROR | `cartesian_prod/reduction/0` | needs manual confirmation | shard1 | 42260539 | mindspore,torch | mindspore: mindspore.ops.functional.cartesian_prod: RuntimeError: ValueError: For 'BroadcastTo', in order to broadcast, each dimension pair must be equal or in... |
| 25 | ERROR | `initial_seed/generic/0` | adapter/input-plan | shard1 | 66260539 | mindspore,torch | mindspore: mindspore.initial_seed: RuntimeError: TypeError: initial_seed() takes 0 positional arguments but 1 was given; torch: torch.initial_seed: TypeError: ... |
| 26 | ERROR | `batch_norm/nn/8` | adapter/input-plan | shard1 | 122260539 | mindspore,torch | mindspore: mindspore.ops.functional.batch_norm: RuntimeError: TypeError: Failed calling BatchNorm with "BatchNorm(is_training=float, epsilon=float, momentum=fl... |
| 27 | ERROR | `dropout2d/nn/4` | backend/env limitation | shard1 | 126260539 | mindspore,torch | mindspore: mindspore.mint.nn.functional.dropout2d: RuntimeError: RuntimeError: Not implement exception ---------------------------------------------------- - C... |
| 28 | DIFF | `relu/nn/2` | DIFF: nonfinite propagation | shard1 | 142260543 | mindspore,paddle,tensorflow,torch | mindspore=[2, 3] [[0.0, inf, 0.0], [0.0, 6.798453330993652, 0.0]]; paddle=[2, 3] [[0.0, inf, 0.0], [0.0, 6.798453330993652, 0.0]]; tensorflow=[2, 3] [[nan, inf, 0.0], [0.0, 6.7984... |
| 29 | DIFF | `argmin/reduction/4` | DIFF: NaN/order semantics | shard1 | 162260561 | jax,numpy,tensorflow | jax=[2] [0, 2]; numpy=[2] [0, 2]; tensorflow=[2] [2, 2] |
| 30 | ERROR | `clip_by_norm/linalg/4` | adapter/input-plan | shard1 | 170260539 | mindspore,tensorflow | mindspore: mindspore.ops.functional.clip_by_norm: RuntimeError: TypeError: clip_by_norm() missing 1 required positional argument: 'max_norm'; tensorflow: tenso... |
| 31 | DIFF | `split/shape/4` | DIFF: adapter normalization | shard1 | 234260539 | mxnet,paddle,tensorflow | mxnet=[{'dtype': 'float32', 'shape': [1, 3], 'value': [[0.6217190623283386,...; paddle=[{'dtype': 'float32', 'shape': [1, 3], 'value': [[0.6217190623283386,...; tensorflow=[{'dtyp... |
| 32 | DIFF | `ceil/generic/2` | DIFF: value semantics | shard1 | 238260540 | mindspore,mxnet,tensorflow | mindspore=[2, 3] [[-0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]; mxnet=[2, 3] [[-0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]; tensorflow=[2, 3] [[-0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]] |
| 33 | DIFF | `lgamma/generic/2` | DIFF: special/edge semantics | shard1 | 246260539 | paddle,tensorflow | paddle=[2, 3] [[inf, inf, 103.2789306640625], [inf, 0.0, 13.8155097961425...; tensorflow=[2, 3] [[inf, inf, 103.97207641601562], [inf, 0.0, 13.815509796142... |
| 34 | ERROR | `cho_solve/linalg/3` | adapter/input-plan | shard1 | 266260539 | mindspore,tensorflow | mindspore: mindspore.ops.functional.cholesky_solve: RuntimeError: TypeError: The primitive[CholeskySolve]'s input arguments[x1, x2] must be all tensor and thos... |
| 35 | ERROR | `lu_solve/linalg/5` | adapter/input-plan | shard1 | 274260539 | jax,mindspore,paddle,scipy,tensorflow | mindspore: mindspore.scipy.linalg.lu_solve: RuntimeError: TypeError: For 'lu_solve', the type of 'lu_matrix' should be Tensor, but got '[[1.7771819 0.50211954]... |
| 36 | ERROR | `conv2d/nn/7` | backend/env limitation | shard1 | 282260539 | mindspore,tensorflow | mindspore: mindspore.mint.nn.functional.conv2d: RuntimeError: RuntimeError: The kernel Conv2DExt unregistered. ------------------------------------------------... |
| 37 | DIFF | `hardswish/generic/1` | DIFF: nonfinite propagation | shard1 | 314260546 | keras,mindspore,tensorflow | keras=[2, 3] [[nan, inf, nan], [-0.06668020784854889, 0.4517600536346435...; mindspore=[2, 3] [[nan, inf, 0.0], [-0.06668020784854889, 0.4517600238323211...; tensorflow=[2, 3] [[n... |
| 38 | DIFF | `relu6/generic/1` | DIFF: nonfinite propagation | shard1 | 322260567 | chainer,keras,mindspore,tensorflow | chainer=[2, 3] [[nan, 6.0, 0.0], [3.024876594543457, 0.0, 4.69653654098510...; keras=[2, 3] [[nan, 6.0, 0.0], [3.024876594543457, 0.0, 4.69653654098510...; mindspore=[2, 3] [[0.0,... |
| 39 | ERROR | `angle/generic/1` | backend/env limitation | shard1 | 386260539 | keras,mindspore | mindspore: mindspore.ops.functional.angle: RuntimeError: TypeError: ---------------------------------------------------- - Kernel select failed: --------------... |
| 40 | DIFF | `argmin/reduction/3` | DIFF: NaN/order semantics | shard1 | 398260548 | jax,keras,mindspore,mxnet | jax=[2] [0, 0]; keras=[2] [2, 0]; mindspore=[2] [0, 0]; +1 libs |
| 41 | DIFF | `ceil/generic/1` | DIFF: value semantics | shard1 | 414260539 | chainer,jax,keras,mindspore | chainer=[2, 3] [[-0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]; jax=[2, 3] [[-0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]; keras=[2, 3] [[-0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]; +1 libs |
| 42 | DIFF | `log10/generic/1` | DIFF: special/edge semantics | shard1 | 490260540 | chainer,jax,keras,mindspore | chainer=[2, 3] [[-inf, -inf, -44.85346984863281], [nan, 0.0, -6.0000004768...; jax=[2, 3] [[-inf, -inf, -inf], [nan, 0.0, -6.0]]; keras=[2, 3] [[-inf, -inf, -45.15449905395508], [... |
| 43 | ERROR | `logaddexp2/generic/2` | adapter/input-plan | shard1 | 494260539 | keras,mindspore | keras: keras.ops.logaddexp2: TypeError: logaddexp2() missing 1 required positional argument: 'x2'; mindspore: mindspore.mint.logaddexp2: RuntimeError: TypeErro... |
| 44 | DIFF | `minimum/generic/2` | DIFF: nonfinite propagation | shard1 | 506260544 | chainer,keras,mindspore,mxnet | chainer=[2, 3] [[nan, 0.0, -inf], [-1.1046050786972046, -0.600511431694030...; keras=[2, 3] [[nan, 0.0, -inf], [-1.1046050786972046, -0.600511431694030...; mindspore=[2, 3] [[-0.0... |
| 45 | ERROR | `slice/generic/3` | adapter/input-plan | shard1 | 546260539 | keras,mindspore | keras: keras.ops.slice: TypeError: slice() missing 2 required positional arguments: 'start_indices' and 'shape'; mindspore: mindspore.ops.slice: RuntimeError: ... |
| 46 | ERROR | `fmin/generic/2` | backend/env limitation | shard1 | 614260539 | jax,mindspore | mindspore: mindspore.ops.functional.fmin: RuntimeError: RuntimeError: ---------------------------------------------------- - Kernel select failed: ------------... |
| 47 | ERROR | `atleast_2d/generic/0` | backend/env limitation | shard2 | 35260539 | jax,mindspore,mxnet,numpy,torch | mindspore: mindspore.numpy.atleast_2d: RuntimeError: SystemError: <built-in method pyboost_reshape of PyCapsule object at 0x7464659862e0> returned a result wit... |
| 48 | ERROR | `get_rng_state/generic/0` | adapter/input-plan | shard2 | 63260539 | mindspore,torch | mindspore: mindspore.get_rng_state: RuntimeError: TypeError: get_rng_state() takes 0 positional arguments but 1 was given; torch: torch.get_rng_state: TypeErro... |
| 49 | ERROR | `set_rng_state/generic/1` | adapter/input-plan | shard2 | 99260539 | mindspore,torch | mindspore: mindspore.set_rng_state: RuntimeError: RuntimeError: State data type must be UInt8, but got Float32 ------------------------------------------------... |
| 50 | ERROR | `adaptive_avg_pool2d/nn/2` | adapter/input-plan | shard2 | 119260539 | mindspore,torch | mindspore: mindspore.mint.nn.functional.adaptive_avg_pool2d: RuntimeError: TypeError: adaptive_avg_pool2d_ext() missing 1 required positional argument: 'output... |
| 51 | ERROR | `elu/nn/3` | backend/env limitation | shard2 | 127260539 | mindspore,paddle,torch | mindspore: mindspore.mint.nn.functional.elu: RuntimeError: RuntimeError: The kernel EluExt unregistered. ---------------------------------------------------- -... |
| 52 | ERROR | `grid_sample/nn/5` | adapter/input-plan | shard2 | 131260539 | mindspore,torch | mindspore: mindspore.mint.nn.functional.grid_sample: RuntimeError: TypeError: grid_sample() missing 1 required positional argument: 'grid'; torch: torch.nn.fun... |
| 53 | ERROR | `hardtanh/nn/4` | backend/env limitation | shard2 | 135260539 | mindspore,paddle,torch | mindspore: mindspore.mint.nn.functional.hardtanh: RuntimeError: RuntimeError: The kernel Hardtanh unregistered. -----------------------------------------------... |
| 54 | ERROR | `max_unpool2d/nn/6` | adapter/input-plan | shard2 | 139260539 | mindspore,torch | mindspore: mindspore.mint.nn.functional.max_unpool2d: RuntimeError: TypeError: max_unpool2d_ext() missing 2 required positional arguments: 'indices' and 'kerne... |
| 55 | ERROR | `relu6/nn/2` | backend/env limitation | shard2 | 143260539 | mindspore,paddle,tensorflow,torch | mindspore: mindspore.mint.nn.functional.relu6: RuntimeError: RuntimeError: The kernel Hardtanh unregistered. --------------------------------------------------... |
| 56 | ERROR | `threshold/nn/4` | backend/env limitation | shard2 | 151260539 | mindspore,torch | mindspore: mindspore.mint.nn.functional.threshold: RuntimeError: RuntimeError: The kernel Threshold unregistered. ---------------------------------------------... |
| 57 | DIFF | `argsort/generic/5` | DIFF: NaN/order semantics | shard2 | 163260555 | numpy,paddle,tensorflow | numpy=[2, 3] [[2, 1, 0], [0, 1, 2]]; paddle=[2, 3] [[2, 1, 0], [0, 1, 2]]; tensorflow=[2, 3] [[0, 2, 1], [0, 1, 2]] |
| 58 | ERROR | `sum/reduction/4` | adapter/input-plan | shard2 | 211260539 | mindspore,tensorflow | mindspore: mindspore.mint.sum: RuntimeError: TypeError: sum() got an unexpected keyword argument 'axis' |
| 59 | DIFF | `sign/generic/2` | DIFF: nonfinite propagation | shard2 | 219260541 | mindspore,mxnet,tensorflow | mindspore=[2, 3] [[nan, nan, nan], [1.0, -1.0, -1.0]]; mxnet=[2, 3] [[0.0, 1.0, -1.0], [1.0, -1.0, -1.0]]; tensorflow=[2, 3] [[nan, 1.0, -1.0], [1.0, -1.0, -1.0]] |
| 60 | ERROR | `sort/generic/4` | backend/env limitation | shard2 | 223260539 | jax,mindspore,mxnet,tensorflow | mindspore: mindspore.mint.sort: RuntimeError: RuntimeError: The kernel SortExt unregistered. ---------------------------------------------------- - C++ Call St... |
| 61 | ERROR | `conv_transpose2d/nn/8` | adapter/input-plan | shard2 | 283260539 | mindspore,tensorflow | mindspore: mindspore.ops.functional.conv_transpose2d: RuntimeError: TypeError: conv_transpose2d() missing 1 required positional argument: 'weight'; tensorflow:... |
| 62 | DIFF | `log_softmax/nn/2` | DIFF: nonfinite propagation | shard2 | 319260539 | chainer,jax,keras,mindspore,scipy,tensorflow | chainer=[2, 3] [[nan, nan, nan], [-1.126882791519165, -1.2128478288650513,...; jax=[2, 3] [[nan, nan, nan], [-1.126882791519165, -1.2128478288650513,...; keras=[2, 3] [[nan, nan, ... |
| 63 | DIFF | `softmax/nn/2` | DIFF: nonfinite propagation | shard2 | 327260560 | chainer,jax,keras,scipy,tensorflow | chainer=[2, 3] [[nan, nan, nan], [3.5585912883107085e-06, 0.00157194235362...; jax=[2, 3] [[nan, nan, nan], [3.5585912883107085e-06, 0.00157194247003...; keras=[2, 3] [[nan, nan, ... |
| 64 | DIFF | `diagflat/generic/2` | DIFF: nonfinite propagation | shard2 | 431260557 | jax,keras,mindspore,mxnet,numpy | jax=[6, 6] [[nan, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, inf, 0.0, 0.0, 0.0, ...; keras=[6, 6] [[nan, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, inf, 0.0, 0.0, 0.0, ...; mindspore=[6, 6] [[nan, nan... |
| 65 | DIFF | `maximum/generic/2` | DIFF: nonfinite propagation | shard2 | 503260539 | chainer,keras,mindspore,mxnet | chainer=[2, 3] [[nan, inf, 0.48599812388420105], [-0.0012984656495973468, ...; keras=[2, 3] [[nan, inf, 0.48599812388420105], [-0.0012984656495973468, ...; mindspore=[2, 3] [[-0.7... |
| 66 | DIFF | `nancumsum/generic/3` | DIFF: nonfinite propagation | shard2 | 511260560 | keras,mindspore | keras=[6] [0.0, 3.4028234663852886e+38, 0.0, -4.2111430168151855, -17...; mindspore=[6] [0.0, inf, nan, nan, nan, nan] |
| 67 | ERROR | `random_integer/random/5` | adapter/input-plan | shard2 | 583260539 | keras,mindspore,paddle | mindspore: UNSUPPORTED: no generic input plan for random_integer/random |
| 68 | ERROR | `histogram_bin_edges/generic/4` | needs manual confirmation | shard2 | 619260539 | jax,mindspore,mxnet,numpy | mxnet: mxnet.numpy.histogram_bin_edges: TimeoutError: external runner response timed out after 30s |
| 69 | ERROR | `atleast_3d/generic/0` | backend/env limitation | shard3 | 36260539 | jax,mindspore,mxnet,numpy,torch | mindspore: mindspore.numpy.atleast_3d: RuntimeError: SystemError: <built-in method pyboost_reshape of PyCapsule object at 0x7ac48fd863a0> returned a result wit... |
| 70 | ERROR | `cdist/generic/4` | adapter/input-plan | shard3 | 44260539 | mindspore,torch | mindspore: mindspore.mint.cdist: RuntimeError: TypeError: cdist() missing 1 required positional argument: 'x2'; torch: torch.cdist: TypeError: cdist() missing ... |
| 71 | ERROR | `solve/linalg/3` | adapter/input-plan | shard3 | 104260539 | mindspore,torch | torch: torch.solve: RuntimeError: This function was deprecated since version 1.9 and is now removed. `torch.solve` is deprecated in favor of `torch.linalg.solv... |
| 72 | ERROR | `adaptive_avg_pool3d/nn/2` | adapter/input-plan | shard3 | 120260539 | mindspore,torch | mindspore: mindspore.mint.nn.functional.adaptive_avg_pool3d: RuntimeError: TypeError: adaptive_avg_pool3d() missing 1 required positional argument: 'output_siz... |
| 73 | ERROR | `group_norm/nn/5` | adapter/input-plan | shard3 | 132260539 | chainer,mindspore,torch | mindspore: mindspore.ops.group_norm: RuntimeError: TypeError: group_norm() missing 1 required positional argument: 'num_groups' |
| 74 | ERROR | `unfold/nn/5` | adapter/input-plan | shard3 | 152260539 | mindspore,torch | mindspore: mindspore.mint.nn.functional.unfold: RuntimeError: TypeError: unfold_ext() missing 1 required positional argument: 'kernel_size' |
| 75 | ERROR | `einsum/generic/1` | backend/env limitation | shard3 | 180260539 | keras,mindspore,paddle,tensorflow | mindspore: mindspore.mint.einsum: RuntimeError: NotImplementedError: einsum only supports Ascend. |
| 76 | DIFF | `logm/linalg/2` | DIFF: numeric/branch semantics | shard3 | 272260768 | scipy,tensorflow | scipy=[2, 2] [[[1.1103631784640262, 1.9516190639709639], [-0.19887627395...; tensorflow=[2, 2] [[[1.1103628873825073, -1.9516187906265259], [-0.1988762021... |
| 77 | ERROR | `conv_transpose3d/nn/8` | adapter/input-plan | shard3 | 284260539 | mindspore,tensorflow | mindspore: mindspore.ops.functional.conv3d_transpose: RuntimeError: TypeError: conv3d_transpose() missing 1 required positional argument: 'weight'; tensorflow:... |
| 78 | ERROR | `uniform_candidate_sampler/random/7` | adapter/input-plan | shard3 | 304260539 | mindspore,tensorflow | mindspore: UNSUPPORTED: no generic input plan for uniform_candidate_sampler/random; tensorflow: UNSUPPORTED: no generic input plan for uniform_candidate_sample... |
| 79 | ERROR | `leaky_relu/nn/2` | adapter/input-plan | shard3 | 316260539 | chainer,jax,keras,mindspore,tensorflow | mindspore: mindspore.ops.leaky_relu: RuntimeError: TypeError: leaky_relu() got an unexpected keyword argument 'negative_slope' |
| 80 | DIFF | `mish/generic/1` | DIFF: nonfinite propagation | shard3 | 320260556 | keras,mindspore,tensorflow | keras=[2, 3] [[nan, inf, nan], [-0.30340149998664856, 0.8650984168052673...; mindspore=[2, 3] [[nan, nan, nan], [-0.30340102314949036, 0.865098237991333,...; tensorflow=[2, 3] [[n... |
| 81 | DIFF | `threshold/generic/3` | DIFF: value semantics | shard3 | 336260540 | keras,mindspore,tensorflow | keras=[2, 3] [[-1.0, -1.0, -1.0], [-1.0, 1.0, 9.999999974752427e-07]]; mindspore=[2, 3] [[-1.0, -1.0, 1.401298464324817e-45], [-1.0, 1.0, 9.9999999...; tensorflow=[2, 3] [[-1.0, -... |
| 82 | DIFF | `clip/generic/3` | DIFF: nonfinite propagation | shard3 | 416260555 | chainer,jax,keras,mindspore | chainer=[2, 3] [[nan, 2.0, -1.0], [0.3003249168395996, -1.0, -0.0947190374...; jax=[2, 3] [[nan, 2.0, -1.0], [0.3003249168395996, -1.0, -0.0947190374...; keras=[2, 3] [[nan, 2.0, ... |
| 83 | ERROR | `count_nonzero/reduction/2` | backend/env limitation | shard3 | 424260539 | keras,mindspore | mindspore: mindspore.mint.count_nonzero: RuntimeError: RuntimeError: The kernel CountNonZero unregistered. ----------------------------------------------------... |
| 84 | DIFF | `log2/generic/1` | DIFF: special/edge semantics | shard3 | 492260545 | chainer,jax,keras,mindspore | chainer=[2, 3] [[-inf, -inf, -149.0], [nan, 0.0, -19.931568145751953]]; jax=[2, 3] [[-inf, -inf, -inf], [nan, 0.0, -19.931568145751953]]; keras=[2, 3] [[-inf, -inf, -150.0], [nan,... |
| 85 | ERROR | `pad/shape/4` | adapter/input-plan | shard3 | 524260539 | keras,mindspore | mindspore: mindspore.ops.pad: RuntimeError: TypeError: For 'pad', the paddings value must be tuple of int or list of int, but got ((1, 1), (1, 1)) |
| 86 | DIFF | `sign/generic/1` | DIFF: value semantics | shard3 | 540260539 | chainer,jax,keras,mindspore | chainer=[2, 3] [[0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]; jax=[2, 3] [[-0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]; keras=[2, 3] [[0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]; +1 libs |
| 87 | DIFF | `tril/generic/2` | DIFF: nonfinite propagation | shard3 | 564260541 | jax,keras,mindspore,mxnet,numpy | jax=[2, 3] [[nan, 0.0, 0.0], [0.12914036214351654, 0.15613389015197754...; keras=[2, 3] [[nan, 0.0, 0.0], [0.12914036214351654, 0.15613389015197754...; mindspore=[2, 3] [[nan, nan... |
| 88 | ERROR | `uniform/random/5` | adapter/input-plan | shard3 | 584260539 | keras,mindspore | mindspore: mindspore.ops.uniform: RuntimeError: TypeError: uniform() missing 2 required positional arguments: 'minval' and 'maxval' |
| 89 | ERROR | `apply_along_axis/generic/3` | needs manual confirmation | shard3 | 588260539 | jax,mindspore,mxnet,numpy | mindspore: mindspore.numpy.apply_along_axis: RuntimeError: TypeError: Failed calling sum with "sum(axis=None, out=None)". The valid calling should be: "Tensor.... |
| 90 | ERROR | `float_power/generic/2` | adapter/input-plan | shard3 | 612260539 | jax,mindspore | jax: jax.numpy.float_power: TypeError: float_power() missing 1 required positional argument: 'y'; mindspore: mindspore.mint.float_power: RuntimeError: TypeErro... |
| 91 | ERROR | `histogramdd/generic/5` | adapter/input-plan | shard3 | 620260550 | jax,mindspore,numpy | numpy: numpy.histogramdd: ValueError: autodetected range of [nan, nan] is not finite |

## Read

- `ERROR` rows are mostly adapter/input-plan gaps; treat them as harness work before filing framework bugs.
- `backend/env limitation` rows are mainly MindSpore CPU kernel gaps or backend-only implementations.
- `DIFF` rows are mostly edge-value behavior from NaN/Inf/subnormal/signed-zero inputs; those are the best candidates for minimization.
