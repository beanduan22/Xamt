from hypothesis import given, settings
from inputs.input_generator import generate_bitwise_not_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bitwise_not
from functions.tf_functions import tf_bitwise_not
from functions.chainer_functions import chainer_bitwise_not

api_functions = {
    "pytorch_bitwise_not": torch_bitwise_not,
    "tensorflow_bitwise_not": tf_bitwise_not,
    "chainer_bitwise_not": chainer_bitwise_not,
}

@given(input_data=generate_bitwise_not_inputs())
@settings(max_examples=100, deadline=None)
def test_bitwise_not_functions(input_data):
    run_test("test_bitwise_not", input_data, api_functions)

if __name__ == "__main__":
    test_bitwise_not_functions()
    finalize_results("test_bitwise_not")
