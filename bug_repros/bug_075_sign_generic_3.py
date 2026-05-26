"""Minimal cross-library repro for bug_075 (sign/generic/3)."""
import importlib
import json
import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

def resolve(qname):
    module_name, _, attr = qname.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def array_value(value):
    if hasattr(value, "resolve_conj") and hasattr(value, "detach"):
        return value.resolve_conj().detach().cpu().numpy()
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    array_attr = getattr(value, "array", None)
    if array_attr is not None:
        return array_attr.get() if hasattr(array_attr, "get") else np.asarray(array_attr)
    return np.asarray(value)


def real_if_close(value):
    arr = array_value(value)
    if arr.dtype.kind == "c" and np.allclose(arr.imag, 0, atol=1e-4, rtol=1e-4):
        return arr.real
    return arr


def encode(value):
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    arr = array_value(value)
    if arr.dtype == object:
        return str(value)
    if arr.dtype.kind == "c":
        stacked = np.stack([arr.real, arr.imag], axis=-1)
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "value": stacked.tolist()}
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "value": arr.tolist()}

def tensor(value, lib, qname):
    if lib == "paddle":
        import paddle

        return paddle.to_tensor(value)
    if lib == "mxnet":
        import mxnet as mx

        if qname.startswith("mxnet.numpy."):
            mx.npx.set_np()
            return mx.np.array(value)
        return mx.nd.array(value)
    return value

def normalize(value):
    return value

BUG_ID = 'bug_075'
KEY = 'sign/generic/3'
EXPECTED_SOURCE = 'reference:paddle'
EXPECTED_LIBS = json.loads(r'''["paddle"]''')
EXPECTED = json.loads(r'''{"dtype": "float32", "shape": [2, 3], "value": [[NaN, 1.0, -1.0], [1.0, -1.0, 1.0]]}''')
WRONG = json.loads(r'''{"mxnet": {"dtype": "float32", "shape": [2, 3], "value": [[0.0, 1.0, -1.0], [1.0, -1.0, 1.0]]}}''')
APIS = json.loads(r'''{"mxnet": "mxnet.ndarray.sign", "paddle": "paddle.sign"}''')

def run_paddle():
    LIB = 'paddle'
    QNAME = 'paddle.sign'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[float("nan"), float("inf"), -float("inf")], [0.1564016342163086, -0.22458967566490173, 0.10585132241249084]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_mxnet():
    LIB = 'mxnet'
    QNAME = 'mxnet.ndarray.sign'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[float("nan"), float("inf"), -float("inf")], [0.1564016342163086, -0.22458967566490173, 0.10585132241249084]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

RUNNERS = {
    'paddle': run_paddle,
    'mxnet': run_mxnet,
}

if __name__ == "__main__":
    print(f"bug_id: {BUG_ID}")
    print(f"key: {KEY}")
    print(f"expected_source: {EXPECTED_SOURCE}")
    print("expected_libs:", ", ".join(EXPECTED_LIBS))
    print("expected:", json.dumps(EXPECTED, sort_keys=True, allow_nan=True))
    print("wrong:")
    for lib, value in WRONG.items():
        print(f"  {lib} ({APIS.get(lib, 'unknown')}): {json.dumps(value, sort_keys=True, allow_nan=True)}")
    print("live_outputs:")
    for lib, runner in RUNNERS.items():
        print(f"  {lib} ({APIS[lib]}): {json.dumps(runner(), sort_keys=True, allow_nan=True)}")
