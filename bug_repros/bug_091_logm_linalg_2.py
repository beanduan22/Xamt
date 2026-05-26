"""Minimal cross-library repro for bug_091 (logm/linalg/2)."""
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
    if lib == "tensorflow":
        import tensorflow as tf

        return tf.constant(value)
    return value

def normalize(value):
    return real_if_close(value)

BUG_ID = 'bug_091'
KEY = 'logm/linalg/2'
EXPECTED_SOURCE = 'reference:scipy'
EXPECTED_LIBS = json.loads(r'''["scipy"]''')
EXPECTED = json.loads(r'''{"dtype": "complex128", "shape": [2, 2], "value": [[[1.1103631784640262, 1.9516190639709639], [-0.1988762739579102, 1.2463638567245867]], [[-0.2973211347022723, 1.863320891720253], [1.2318952214097851, 1.189973932819967]]]}''')
WRONG = json.loads(r'''{"tensorflow": {"dtype": "complex64", "shape": [2, 2], "value": [[[1.1103628873825073, -1.9516187906265259], [-0.19887620210647583, -1.246363639831543]], [[-0.29732105135917664, -1.8633205890655518], [1.231894850730896, -1.1899737119674683]]]}}''')
APIS = json.loads(r'''{"scipy": "scipy.linalg.logm", "tensorflow": "tensorflow.linalg.logm"}''')

def run_scipy():
    LIB = 'scipy'
    QNAME = 'scipy.linalg.logm'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[0.010266004130244255, -2.640225887298584], [-3.9471523761749268, 1.6236913204193115]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_tensorflow():
    LIB = 'tensorflow'
    QNAME = 'tensorflow.linalg.logm'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[complex(0.010266004130244255, 0.0), complex(-2.640225887298584, 0.0)], [complex(-3.9471523761749268, 0.0), complex(1.6236913204193115, 0.0)]], dtype=np.complex64), LIB, QNAME))
    return encode(normalize(result))

RUNNERS = {
    'scipy': run_scipy,
    'tensorflow': run_tensorflow,
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
