"""Minimal cross-library repro for bug_085 (unique/generic/component/1)."""
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
    if lib in {"numpy", "scipy", "chainer"}:
        return value
    if lib == "mindspore":
        import mindspore as ms

        return ms.Tensor(value)
    return value

def normalize(value):
    if isinstance(value, (list, tuple)):
        value = value[0]
    arr = np.asarray(real_if_close(value))
    if arr.ndim == 1:
        order = np.lexsort((arr.imag, arr.real)) if arr.dtype.kind == "c" else np.argsort(arr, kind="stable")
        arr = arr[order]
    return arr

BUG_ID = 'bug_085'
KEY = 'unique/generic/component/1'
EXPECTED_SOURCE = 'majority'
EXPECTED_LIBS = json.loads(r'''["jax", "mxnet", "numpy", "paddle", "torch"]''')
EXPECTED = json.loads(r'''{"dtype": "float32", "shape": [6], "value": [-11.842028617858887, -10.728705406188965, 1.2593802213668823, 7.331770420074463, 9.312121391296387, 11.105917930603027]}''')
WRONG = json.loads(r'''{"mindspore": {"dtype": "float32", "shape": [6], "value": [7.331770420074463, 1.2593802213668823, 11.105917930603027, -10.728705406188965, 9.312121391296387, -11.842028617858887]}}''')
APIS = json.loads(r'''{"mindspore": "mindspore.numpy.unique", "numpy": "numpy.unique"}''')

def run_numpy():
    LIB = 'numpy'
    QNAME = 'numpy.unique'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[float("nan"), float("inf"), -float("inf")], [0.0, -0.0, 1.0]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_mindspore():
    LIB = 'mindspore'
    QNAME = 'mindspore.numpy.unique'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[float("nan"), float("inf"), -float("inf")], [0.0, -0.0, 1.0]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

RUNNERS = {
    'numpy': run_numpy,
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
