from hypothesis import given, settings
from inputs.input_generator import generate_bitwise_right_shift_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bitwise_right_shift
from functions.tf_functions import tf_bitwise_right_shift
from functions.jax_functions import jax_bitwise_right_shift
from functions.chainer_functions import chainer_bitwise_right_shift
from functions.keras_functions import keras_bitwise_right_shift

api_functions = {
    "pytorch_bitwise_right_shift": torch_bitwise_right_shift,
    "tensorflow_bitwise_right_shift": tf_bitwise_right_shift,
    "jax_bitwise_right_shift": jax_bitwise_right_shift,
    "chainer_bitwise_right_shift": chainer_bitwise_right_shift,
    "keras_bitwise_right_shift": keras_bitwise_right_shift,
}

@given(input_data=generate_bitwise_right_shift_inputs())
@settings(max_examples=100, deadline=None)
def test_bitwise_right_shift_functions(input_data):
    run_test("test_bitwise_right_shift", input_data, api_functions)

if __name__ == "__main__":
    test_bitwise_right_shift_functions()
    finalize_results("test_bitwise_right_shift")
