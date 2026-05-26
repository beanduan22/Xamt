"""Minimal cross-library repro for bug_096 (logm/linalg/component/1)."""
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

BUG_ID = 'bug_096'
KEY = 'logm/linalg/component/1'
EXPECTED_SOURCE = 'reference:scipy'
EXPECTED_LIBS = json.loads(r'''["scipy"]''')
EXPECTED = json.loads(r'''{"dtype": "float64", "shape": [2, 2], "value": [[0.648307993840645, -0.14794227141650157], [-0.20096137966968625, 0.8326601466693216]]}''')
WRONG = json.loads(r'''{"tensorflow": {"dtype": "complex64", "shape": [2, 2], "value": [[[0.6483079195022583, 0.0], [-0.14794224500656128, 0.0]], [[-0.2009613960981369, 0.0], [0.8326600193977356, 0.0]]]}}''')
APIS = json.loads(r'''{"scipy": "scipy.linalg.logm", "tensorflow": "tensorflow.linalg.logm"}''')

def run_scipy():
    LIB = 'scipy'
    QNAME = 'scipy.linalg.logm'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[0.020969873294234276, -2.05605411529541], [-2.3816792964935303, 3.90266752243042]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_tensorflow():
    LIB = 'tensorflow'
    QNAME = 'tensorflow.linalg.logm'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[complex(0.020969873294234276, 0.0), complex(-2.05605411529541, 0.0)], [complex(-2.3816792964935303, 0.0), complex(3.90266752243042, 0.0)]], dtype=np.complex64), LIB, QNAME))
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
