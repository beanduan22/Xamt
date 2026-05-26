"""Minimal cross-library repro for bug_034 (msort/generic/1)."""
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
    if lib == "mindspore":
        import mindspore as ms

        return ms.Tensor(value)
    if lib == "mxnet":
        import mxnet as mx

        if qname.startswith("mxnet.numpy."):
            mx.npx.set_np()
            return mx.np.array(value)
        return mx.nd.array(value)
    return value

def normalize(value):
    return value

BUG_ID = 'bug_034'
KEY = 'msort/generic/1'
EXPECTED_SOURCE = 'reference:mxnet'
EXPECTED_LIBS = json.loads(r'''["mxnet"]''')
EXPECTED = json.loads(r'''{"dtype": "float32", "shape": [2, 3], "value": [[-0.6830241084098816, -0.30688419938087463, -Infinity], [NaN, Infinity, 3.4525134563446045]]}''')
WRONG = json.loads(r'''{"mindspore": {"dtype": "float32", "shape": [2, 3], "value": [[NaN, -0.30688419938087463, -Infinity], [-0.6830241084098816, Infinity, 3.4525134563446045]]}}''')
APIS = json.loads(r'''{"mindspore": "mindspore.ops.function.msort", "mxnet": "mxnet.numpy.msort"}''')

def run_mxnet():
    LIB = 'mxnet'
    QNAME = 'mxnet.numpy.msort'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[float("nan"), float("inf"), -float("inf")], [-0.6830241084098816, -0.30688419938087463, 3.4525134563446045]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_mindspore():
    LIB = 'mindspore'
    QNAME = 'mindspore.ops.function.msort'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[float("nan"), float("inf"), -float("inf")], [-0.6830241084098816, -0.30688419938087463, 3.4525134563446045]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

RUNNERS = {
    'mxnet': run_mxnet,
    'mindspore': run_mindspore,
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
