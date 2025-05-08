from hypothesis import given, settings
from inputs.input_generator import generate_nn_fractional_max_pool3d_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_nn_fractional_max_pool3d
from functions.tf_functions import tf_nn_fractional_max_pool3d
from functions.mxnet_functions import mxnet_nn_fractional_max_pool3d
from functions.jax_functions import jax_nn_fractional_max_pool3d

api_functions = {
    "pytorch_nn_fractional_max_pool3d": torch_nn_fractional_max_pool3d,
    "tensorflow_nn_fractional_max_pool3d": tf_nn_fractional_max_pool3d,
    "mxnet_nn_fractional_max_pool3d": mxnet_nn_fractional_max_pool3d,
    "jax_nn_fractional_max_pool3d": jax_nn_fractional_max_pool3d,
}

@given(input_data=generate_nn_fractional_max_pool3d_input())
@settings(max_examples=100, deadline=None)
def test_nn_fractional_max_pool3d(input_data):
    run_test("test_nn_fractional_max_pool3d", input_data, api_functions)

if __name__ == "__main__":
    test_nn_fractional_max_pool3d()
    finalize_results("test_nn_fractional_max_pool3d")
