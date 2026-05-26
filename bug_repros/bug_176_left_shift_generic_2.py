"""Minimal cross-library repro for bug_176 (left_shift/generic/2)."""
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
    if lib == "jax":
        import jax.numpy as jnp

        return jnp.asarray(value)
    if lib == "keras":
        import keras

        return keras.ops.convert_to_tensor(value)
    if lib == "mindspore":
        import mindspore as ms

        return ms.Tensor(value)
    return value

def normalize(value):
    return value

BUG_ID = 'bug_176'
KEY = 'left_shift/generic/2'
EXPECTED_SOURCE = 'reference:jax'
EXPECTED_LIBS = json.loads(r'''["jax"]''')
EXPECTED = json.loads(r'''{"dtype": "int32", "shape": [3], "value": [0, -4, 0]}''')
WRONG = json.loads(r'''{"keras": {"dtype": "int64", "shape": [3], "value": [0, -4, 2]}, "mindspore": {"dtype": "int64", "shape": [3], "value": [0, -4, -9223372036854775808]}}''')
APIS = json.loads(r'''{"jax": "jax.numpy.bitwise_left_shift", "keras": "keras.ops.bitwise_left_shift", "mindspore": "mindspore.ops.bitwise_left_shift"}''')

def run_jax():
    LIB = 'jax'
    QNAME = 'jax.numpy.bitwise_left_shift'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([0, -1, 2], dtype=np.int64), LIB, QNAME), tensor(np.array([0, 2, -2], dtype=np.int64), LIB, QNAME))
    return encode(normalize(result))

def run_keras():
    LIB = 'keras'
    QNAME = 'keras.ops.bitwise_left_shift'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([0, -1, 2], dtype=np.int64), LIB, QNAME), tensor(np.array([0, 2, -2], dtype=np.int64), LIB, QNAME))
    return encode(normalize(result))

def run_mindspore():
    LIB = 'mindspore'
    QNAME = 'mindspore.ops.bitwise_left_shift'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([0, -1, 2], dtype=np.int64), LIB, QNAME), tensor(np.array([0, 2, -2], dtype=np.int64), LIB, QNAME))
    return encode(normalize(result))

RUNNERS = {
    'jax': run_jax,
    'keras': run_keras,
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
