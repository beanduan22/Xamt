"""Minimal cross-library repro for bug_097 (lstsq/generic/component/1)."""
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
    if lib == "torch":
        import torch

        return torch.tensor(value)
    if lib == "keras":
        import keras

        return keras.ops.convert_to_tensor(value)
    return value

def normalize(value):
    if hasattr(value, "solution"):
        return value.solution
    if isinstance(value, (list, tuple)):
        return value[0]
    return value

BUG_ID = 'bug_097'
KEY = 'lstsq/generic/component/1'
EXPECTED_SOURCE = 'majority'
EXPECTED_LIBS = json.loads(r'''["keras", "mindspore", "torch"]''')
EXPECTED = json.loads(r'''{"dtype": "float32", "shape": [2, 1], "value": [[-32.44655227661133], [-15.117897033691406]]}''')
WRONG = json.loads(r'''{}''')
APIS = json.loads(r'''{"keras": "keras.ops.lstsq", "torch": "torch.lstsq"}''')

def run_torch():
    LIB = 'torch'
    QNAME = 'torch.lstsq'
    import torch
    result = torch.linalg.lstsq(tensor(np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32), LIB, QNAME), tensor(np.array([[1.0], [2.0]], dtype=np.float32), LIB, QNAME)).solution
    return encode(normalize(result))

def run_keras():
    LIB = 'keras'
    QNAME = 'keras.ops.lstsq'
    fn = resolve(QNAME)
    result = fn(tensor(np.array([[-0.2574699819087982, -0.3612658679485321], [1.787086844444275, 2.505218744277954]], dtype=np.float32), LIB, QNAME), tensor(np.array([[5.359724521636963], [2.811603546142578]], dtype=np.float32), LIB, QNAME))
    return encode(normalize(result))

RUNNERS = {
    'torch': run_torch,
    'keras': run_keras,
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
