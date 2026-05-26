"""Minimal cross-library repro for bug_098 (lstsq/linalg/component/1)."""
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
    if hasattr(value, "solution"):
        return value.solution
    if isinstance(value, (list, tuple)):
        return value[0]
    return value

BUG_ID = 'bug_098'
KEY = 'lstsq/linalg/component/1'
EXPECTED_SOURCE = 'majority'
EXPECTED_LIBS = json.loads(r'''["jax", "mindspore", "mxnet", "numpy", "paddle", "scipy", "tensorflow", "torch"]''')
EXPECTED = json.loads(r'''{"dtype": "float32", "shape": [2, 1], "value": [[-27.536666870117188], [45.963077545166016]]}''')
WRONG = json.loads(r'''{}''')
APIS = json.loads(r'''{"jax": "jax.numpy.linalg.lstsq", "numpy": "numpy.linalg.lstsq"}''')

def run_numpy():
    LIB = 'numpy'
    QNAME = 'numpy.linalg.lstsq'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[-1.746012806892395, 5.429816722869873], [-0.9095338582992554, 2.820603132247925]], dtype=np.float32), LIB, QNAME), tensor(np.array([[0.04647109657526016], [3.432987928390503]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_jax():
    LIB = 'jax'
    QNAME = 'jax.numpy.linalg.lstsq'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[-1.746012806892395, 5.429816722869873], [-0.9095338582992554, 2.820603132247925]], dtype=np.float32), LIB, QNAME), tensor(np.array([[0.04647109657526016], [3.432987928390503]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

RUNNERS = {
    'numpy': run_numpy,
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
