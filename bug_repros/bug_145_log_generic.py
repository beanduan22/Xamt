"""Minimal cross-library repro for bug_145 (log/generic)."""
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
    if lib == "tensorflow":
        import tensorflow as tf

        return tf.constant(value)
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

BUG_ID = 'bug_145'
KEY = 'log/generic'
EXPECTED_SOURCE = 'majority'
EXPECTED_LIBS = json.loads(r'''["chainer", "jax", "mxnet", "paddle", "torch"]''')
EXPECTED = json.loads(r'''{"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -103.2789306640625, -13.815510749816895], [-0.6931471824645996, 0.0, Infinity]]}''')
WRONG = json.loads(r'''{"keras": {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -13.815510749816895], [-0.6931471824645996, 0.0, Infinity]]}, "mindspore": {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -88.02967071533203, -13.815506935119629], [-0.6931470036506653, 0.0, Infinity]]}, "tensorflow": {"dtype": "float32", "shape": [2, 3], "value": [[-Infinity, -Infinity, -13.815510749816895], [-0.6931471824645996, 0.0, Infinity]]}}''')
APIS = json.loads(r'''{"jax": "jax.numpy.log", "keras": "keras.ops.log", "mindspore": "mindspore.numpy.log", "tensorflow": "tensorflow.math.log"}''')

def run_jax():
    LIB = 'jax'
    QNAME = 'jax.numpy.log'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[0.0, 1.401298464324817e-45, 1.0], [float("inf"), 1.0000000031710769e-30, 4.0]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_keras():
    LIB = 'keras'
    QNAME = 'keras.ops.log'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[0.0, 1.401298464324817e-45, 1.0], [float("inf"), 1.0000000031710769e-30, 4.0]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_mindspore():
    LIB = 'mindspore'
    QNAME = 'mindspore.numpy.log'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[0.0, 1.401298464324817e-45, 1.0], [float("inf"), 1.0000000031710769e-30, 4.0]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_tensorflow():
    LIB = 'tensorflow'
    QNAME = 'tensorflow.math.log'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[0.0, 1.401298464324817e-45, 1.0], [float("inf"), 1.0000000031710769e-30, 4.0]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

RUNNERS = {
    'jax': run_jax,
    'keras': run_keras,
    'mindspore': run_mindspore,
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
