from hypothesis import given, settings
from inputs.input_generator import generate_conv2d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_conv2d
from functions.tf_functions import tf_conv2d
from functions.chainer_functions import chainer_conv2d

api_functions = {
    "pytorch_conv2d": torch_conv2d,
    "tensorflow_conv2d": tf_conv2d,
    "chainer_conv2d": chainer_conv2d,
}

@given(input_data=generate_conv2d_inputs())
@settings(max_examples=100, deadline=None)
def test_conv2d_functions(input_data):
    run_test("test_conv2d", input_data, api_functions)

if __name__ == "__main__":
    test_conv2d_functions()
    finalize_results("test_conv2d")
