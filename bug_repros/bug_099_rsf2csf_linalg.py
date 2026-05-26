"""Minimal cross-library repro for bug_099 (rsf2csf/linalg)."""
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

BUG_ID = 'bug_099'
KEY = 'rsf2csf/linalg'
EXPECTED_SOURCE = 'reference:scipy'
EXPECTED_LIBS = json.loads(r'''["scipy"]''')
EXPECTED = json.loads(r'''[{"dtype": "complex64", "shape": [2, 2], "value": [[[3.251811981201172, 1.6166623830795288], [-3.5150370597839355, 6.817754183430225e-08]], [[0.0, 0.0], [3.2518110275268555, -1.6166623830795288]]]}, {"dtype": "complex64", "shape": [2, 2], "value": [[[0.260574072599411, 0.34882885217666626], [-0.8944792151451111, -0.10161858797073364]], [[0.8944792747497559, -0.10161858797073364], [0.260574072599411, -0.34882885217666626]]]}]''')
WRONG = json.loads(r'''{"jax": [{"dtype": "complex64", "shape": [2, 2], "value": [[[3.2514841556549072, 1.6160972118377686], [-3.514176845550537, 0.000377655029296875]], [[0.0, 0.0], [3.251483917236328, -1.6163119077682495]]]}, {"dtype": "complex64", "shape": [2, 2], "value": [[[0.2606593072414398, 0.3487358093261719], [-0.8943385481834412, -0.10164070129394531]], [[0.8943385481834412, -0.10164070129394531], [0.2606593072414398, -0.3487358093261719]]]}]}''')
APIS = json.loads(r'''{"jax": "jax.scipy.linalg.rsf2csf", "scipy": "scipy.linalg.rsf2csf"}''')

def run_scipy():
    LIB = 'scipy'
    QNAME = 'scipy.linalg.rsf2csf'
    fn = resolve(QNAME)
    result = fn(np.array([[3.2518117427825928, -0.6304657459259033], [4.145503044128418, 3.2518117427825928]], dtype=np.float32), np.array([[0.9600910544395447, 0.2796875834465027], [-0.2796875834465027, 0.9600910544395447]], dtype=np.float32))
    return encode(normalize(result))

def run_jax():
    LIB = 'jax'
    QNAME = 'jax.scipy.linalg.rsf2csf'
    fn = resolve(QNAME)
    result = fn(np.array([[3.2518117427825928, -0.6304657459259033], [4.145503044128418, 3.2518117427825928]], dtype=np.float32), np.array([[0.9600910544395447, 0.2796875834465027], [-0.2796875834465027, 0.9600910544395447]], dtype=np.float32))
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
