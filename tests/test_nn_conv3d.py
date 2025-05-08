from hypothesis import given, strategies as st, settings
from inputs.input_generator import generate_nn_conv3d_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_conv3d
from functions.tf_functions import nn_tf_conv3d
from functions.keras_functions import nn_keras_conv3d
from functions.chainer_functions import nn_chainer_conv3d
from functions.jax_functions import nn_jax_conv3d

api_functions = {
    "pytorch_nn_conv3d": nn_torch_conv3d,
    "tensorflow_nn_conv3d": nn_tf_conv3d,
    "keras_nn_conv3d": nn_keras_conv3d,
    "chainer_nn_conv3d": nn_chainer_conv3d,
    "jax_nn_conv3d": nn_jax_conv3d,
}

@given(input_data=generate_nn_conv3d_input())
@settings(max_examples=10, deadline=None)
def test_nn_conv3d_functions(input_data):
    run_test("test_nn_conv3d", input_data, api_functions)

if __name__ == "__main__":
    test_nn_conv3d_functions()
    finalize_results("test_nn_conv3d")
