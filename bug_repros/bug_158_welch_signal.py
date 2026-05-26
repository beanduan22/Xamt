"""Minimal cross-library repro for bug_158 (welch/signal)."""
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
    if lib == "jax":
        import jax.numpy as jnp

        return jnp.asarray(value)
    return value

def normalize(value):
    return value

BUG_ID = 'bug_158'
KEY = 'welch/signal'
EXPECTED_SOURCE = 'reference:scipy'
EXPECTED_LIBS = json.loads(r'''["scipy"]''')
EXPECTED = json.loads(r'''[{"dtype": "float64", "shape": [3], "value": [0.0, 0.25, 0.5]}, {"dtype": "float32", "shape": [3], "value": [Infinity, Infinity, Infinity]}]''')
WRONG = json.loads(r'''{"jax": [{"dtype": "float32", "shape": [3], "value": [0.0, 0.25, 0.5]}, {"dtype": "float32", "shape": [3], "value": [Infinity, NaN, Infinity]}]}''')
APIS = json.loads(r'''{"jax": "jax.scipy.signal.welch", "scipy": "scipy.signal.welch"}''')

def run_scipy():
    LIB = 'scipy'
    QNAME = 'scipy.signal.welch'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([9.999999680285692e+37, -9.999999680285692e+37, 1.401298464324817e-45, -0.0], dtype=np.float32), LIB, QNAME), nperseg=4)
    return encode(normalize(result))

def run_jax():
    LIB = 'jax'
    QNAME = 'jax.scipy.signal.welch'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([9.999999680285692e+37, -9.999999680285692e+37, 1.401298464324817e-45, -0.0], dtype=np.float32), LIB, QNAME), nperseg=4)
    return encode(normalize(result))

RUNNERS = {
    'scipy': run_scipy,
    'jax': run_jax,
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
