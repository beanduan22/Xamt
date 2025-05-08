from hypothesis import given, settings
from inputs.input_generator import generate_bitwise_or_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bitwise_or
from functions.tf_functions import tf_bitwise_or
from functions.chainer_functions import chainer_bitwise_or

api_functions = {
    "pytorch_bitwise_or": torch_bitwise_or,
    "tensorflow_bitwise_or": tf_bitwise_or,
    "chainer_bitwise_or": chainer_bitwise_or,
}

@given(input_data=generate_bitwise_or_inputs())
@settings(max_examples=100, deadline=None)
def test_bitwise_or_functions(input_data):
    run_test("test_bitwise_or", input_data, api_functions)

if __name__ == "__main__":
    test_bitwise_or_functions()
    finalize_results("test_bitwise_or")
