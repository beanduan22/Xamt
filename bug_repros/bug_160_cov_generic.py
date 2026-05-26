"""Minimal cross-library repro for bug_160 (cov/generic)."""
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
    if lib == "torch":
        import torch

        return torch.tensor(value)
    if lib == "jax":
        import jax.numpy as jnp

        return jnp.asarray(value)
    return value

def normalize(value):
    return value

BUG_ID = 'bug_160'
KEY = 'cov/generic'
EXPECTED_SOURCE = 'reference:numpy'
EXPECTED_LIBS = json.loads(r'''["numpy"]''')
EXPECTED = json.loads(r'''{"dtype": "float64", "shape": [2, 2], "value": [[9.999999360571395e+75, -7.006492097616501e-08], [-7.006492097616501e-08, 6.545457953730302e-91]]}''')
WRONG = json.loads(r'''{"jax": {"dtype": "float32", "shape": [2, 2], "value": [[Infinity, 0.0], [0.0, 0.0]]}, "torch": {"dtype": "float32", "shape": [2, 2], "value": [[Infinity, -7.006492097616501e-08], [-7.006492097616501e-08, 0.0]]}}''')
APIS = json.loads(r'''{"jax": "jax.numpy.cov", "numpy": "numpy.cov", "torch": "torch.cov"}''')

def run_numpy():
    LIB = 'numpy'
    QNAME = 'numpy.cov'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[9.999999680285692e+37, -9.999999680285692e+37, 1.401298464324817e-45], [-1.401298464324817e-45, 0.0, -0.0]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_jax():
    LIB = 'jax'
    QNAME = 'jax.numpy.cov'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[9.999999680285692e+37, -9.999999680285692e+37, 1.401298464324817e-45], [-1.401298464324817e-45, 0.0, -0.0]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_torch():
    LIB = 'torch'
    QNAME = 'torch.cov'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[9.999999680285692e+37, -9.999999680285692e+37, 1.401298464324817e-45], [-1.401298464324817e-45, 0.0, -0.0]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

RUNNERS = {
    'numpy': run_numpy,
    'jax': run_jax,
    'torch': run_torch,
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
