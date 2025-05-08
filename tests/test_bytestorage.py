from hypothesis import given, settings
import hypothesis.strategies as st
from inputs.input_generator import generate_byte_storage_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_byte_storage
from functions.tf_functions import tf_byte_storage
from functions.jax_functions import jax_byte_storage
from functions.chainer_functions import chainer_byte_storage
from functions.keras_functions import keras_byte_storage

api_functions = {
    "pytorch_byte_storage": torch_byte_storage,
    "tensorflow_byte_storage": tf_byte_storage,
    "jax_byte_storage": jax_byte_storage,
    "chainer_byte_storage": chainer_byte_storage,
    "keras_byte_storage": keras_byte_storage,
}

@given(input_data=generate_byte_storage_inputs())
@settings(max_examples=100, deadline=None)
def test_byte_storage_functions(input_data):
    run_test("test_byte_storage", input_data, api_functions)

if __name__ == "__main__":
    test_byte_storage_functions()
    finalize_results("test_byte_storage")
