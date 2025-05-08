from hypothesis import given, settings
from inputs.input_generator import generate_bitwise_xor_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bitwise_xor
from functions.tf_functions import tf_bitwise_xor
from functions.jax_functions import jax_bitwise_xor
from functions.chainer_functions import chainer_bitwise_xor
from functions.keras_functions import keras_bitwise_xor

api_functions = {
    "pytorch_bitwise_xor": torch_bitwise_xor,
    "tensorflow_bitwise_xor": tf_bitwise_xor,
    "jax_bitwise_xor": jax_bitwise_xor,
    "chainer_bitwise_xor": chainer_bitwise_xor,
    "keras_bitwise_xor": keras_bitwise_xor,
}

@given(input_data=generate_bitwise_xor_inputs())
@settings(max_examples=100, deadline=None)
def test_bitwise_xor_functions(input_data):
    run_test("test_bitwise_xor", input_data, api_functions)

if __name__ == "__main__":
    test_bitwise_xor_functions()
    finalize_results("test_bitwise_xor")
