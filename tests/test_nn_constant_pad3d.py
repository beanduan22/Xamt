from hypothesis import given, strategies as st, settings
from inputs.input_generator import generate_nn_constant_pad3d_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_constant_pad3d
from functions.tf_functions import nn_tf_constant_pad3d
from functions.keras_functions import nn_keras_constant_pad3d
from functions.chainer_functions import nn_chainer_constant_pad3d
from functions.jax_functions import nn_jax_constant_pad3d

api_functions = {
    "pytorch_nn_constant_pad3d": nn_torch_constant_pad3d,
    "tensorflow_nn_constant_pad3d": nn_tf_constant_pad3d,
    "keras_nn_constant_pad3d": nn_keras_constant_pad3d,
    "chainer_nn_constant_pad3d": nn_chainer_constant_pad3d,
    "jax_nn_constant_pad3d": nn_jax_constant_pad3d,
}

@given(input_data=generate_nn_constant_pad3d_input())
@settings(max_examples=10, deadline=None)
def test_nn_constant_pad3d_functions(input_data):
    run_test("test_nn_constant_pad3d", input_data, api_functions)

if __name__ == "__main__":
    test_nn_constant_pad3d_functions()
    finalize_results("test_nn_constant_pad3d")
