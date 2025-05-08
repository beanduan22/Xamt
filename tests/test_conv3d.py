from hypothesis import given, settings
from inputs.input_generator import generate_conv3d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_conv3d
from functions.tf_functions import tf_conv3d
from functions.chainer_functions import chainer_conv3d

api_functions = {
    "pytorch_conv3d": torch_conv3d,
    "tensorflow_conv3d": tf_conv3d,
    "chainer_conv3d": chainer_conv3d,
}

@given(input_data=generate_conv3d_inputs())
@settings(max_examples=100, deadline=None)
def test_conv3d_functions(input_data):
    run_test("test_conv3d", input_data, api_functions)

if __name__ == "__main__":
    test_conv3d_functions()
    finalize_results("test_conv3d")
