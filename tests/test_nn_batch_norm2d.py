from hypothesis import given, strategies as st, settings
from inputs.input_generator import generate_nn_batch_norm2d_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_batch_norm2d
from functions.tf_functions import nn_tf_batch_norm2d
from functions.keras_functions import nn_keras_batch_norm2d
from functions.chainer_functions import nn_chainer_batch_norm2d
from functions.jax_functions import nn_jax_batch_norm2d

api_functions = {
    "pytorch_nn_batch_norm2d": nn_torch_batch_norm2d,
    "tensorflow_nn_batch_norm2d": nn_tf_batch_norm2d,
    "keras_nn_batch_norm2d": nn_keras_batch_norm2d,
    "chainer_nn_batch_norm2d": nn_chainer_batch_norm2d,
    "jax_nn_batch_norm2d": nn_jax_batch_norm2d,
}

@given(input_data=generate_nn_batch_norm2d_input())
@settings(max_examples=10, deadline=None)
def test_nn_batch_norm2d_functions(input_data):
    run_test("test_nn_batch_norm2d", input_data, api_functions)

if __name__ == "__main__":
    test_nn_batch_norm2d_functions()
    finalize_results("test_nn_batch_norm2d")
