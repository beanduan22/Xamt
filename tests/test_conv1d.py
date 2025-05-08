from hypothesis import given, settings
from inputs.input_generator import generate_conv1d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_conv1d
from functions.tf_functions import tf_conv1d
from functions.chainer_functions import chainer_conv1d

api_functions = {
    "pytorch_conv1d": torch_conv1d,
    "tensorflow_conv1d": tf_conv1d,
    "chainer_conv1d": chainer_conv1d,
}

@given(input_data=generate_conv1d_inputs())
@settings(max_examples=100, deadline=None)
def test_conv1d_functions(input_data):
    run_test("test_conv1d", input_data, api_functions)

if __name__ == "__main__":
    test_conv1d_functions()
    finalize_results("test_conv1d")
