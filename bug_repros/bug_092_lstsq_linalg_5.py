"""Minimal cross-library repro for bug_092 (lstsq/linalg/5)."""
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
    if lib == "paddle":
        import paddle

        return paddle.to_tensor(value)
    return value

def normalize(value):
    if hasattr(value, "solution"):
        return value.solution
    if isinstance(value, (list, tuple)):
        return value[0]
    return value

BUG_ID = 'bug_092'
KEY = 'lstsq/linalg/5'
EXPECTED_SOURCE = 'reference:tensorflow'
EXPECTED_LIBS = json.loads(r'''["tensorflow"]''')
EXPECTED = json.loads(r'''{"dtype": "float32", "shape": [2, 1], "value": [[-1099.765869140625], [285.66302490234375]]}''')
WRONG = json.loads(r'''{"paddle": {"dtype": "float32", "shape": [2, 1], "value": [[-1134.48486328125], [294.6731872558594]]}}''')
APIS = json.loads(r'''{"paddle": "paddle.linalg.lstsq", "tensorflow": "tensorflow.linalg.lstsq"}''')

def run_tensorflow():
    LIB = 'tensorflow'
    QNAME = 'tensorflow.linalg.lstsq'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[5.760176181793213, 22.182636260986328], [1.509578824043274, 5.867371559143066]], dtype=np.float32), LIB, QNAME), tensor(np.array([[1.7950254678726196], [16.362768173217773]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

def run_paddle():
    LIB = 'paddle'
    QNAME = 'paddle.linalg.lstsq'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[5.760176181793213, 22.182636260986328], [1.509578824043274, 5.867371559143066]], dtype=np.float32), LIB, QNAME), tensor(np.array([[1.7950254678726196], [16.362768173217773]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

RUNNERS = {
    'tensorflow': run_tensorflow,
    'paddle': run_paddle,
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
