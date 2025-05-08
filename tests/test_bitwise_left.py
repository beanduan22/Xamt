from hypothesis import given, settings
from inputs.input_generator import generate_bitwise_left_shift_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bitwise_left_shift
from functions.tf_functions import tf_bitwise_left_shift
from functions.chainer_functions import chainer_bitwise_left_shift

api_functions = {
    "pytorch_bitwise_left_shift": torch_bitwise_left_shift,
    "tensorflow_bitwise_left_shift": tf_bitwise_left_shift,
    "chainer_bitwise_left_shift": chainer_bitwise_left_shift,
}

@given(input_data=generate_bitwise_left_shift_inputs())
@settings(max_examples=100, deadline=None)
def test_bitwise_left_shift_functions(input_data):
    run_test("test_bitwise_left_shift", input_data, api_functions)

if __name__ == "__main__":
    test_bitwise_left_shift_functions()
    finalize_results("test_bitwise_left_shift")
