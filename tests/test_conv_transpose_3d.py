from hypothesis import given, settings
from inputs.input_generator import generate_conv_transpose_3d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_conv_transpose_3d
from functions.tf_functions import tf_conv_transpose_3d

api_functions = {
    "pytorch_conv_transpose_3d": torch_conv_transpose_3d,
    "tensorflow_conv_transpose_3d": tf_conv_transpose_3d,
}

@given(input_data=generate_conv_transpose_3d_inputs())
@settings(max_examples=100, deadline=None)
def test_conv_transpose_3d_functions(input_data):
    run_test("test_conv_transpose_3d", input_data, api_functions)

if __name__ == "__main__":
    test_conv_transpose_3d_functions()
    finalize_results("test_conv_transpose_3d")
