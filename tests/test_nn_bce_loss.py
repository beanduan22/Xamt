from hypothesis import given, strategies as st, settings
from inputs.input_generator import generate_nn_bce_loss_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_bce_loss
from functions.tf_functions import nn_tf_binary_crossentropy
from functions.keras_functions import nn_keras_binary_crossentropy
from functions.chainer_functions import nn_chainer_binary_cross_entropy
from functions.jax_functions import nn_jax_binary_crossentropy

api_functions = {
    "pytorch_nn_bce_loss": nn_torch_bce_loss,
    "tensorflow_nn_binary_crossentropy": nn_tf_binary_crossentropy,
    "keras_nn_binary_crossentropy": nn_keras_binary_crossentropy,
    "chainer_nn_binary_cross_entropy": nn_chainer_binary_cross_entropy,
    "jax_nn_binary_crossentropy": nn_jax_binary_crossentropy,
}

@given(input_data=generate_nn_bce_loss_input())
@settings(max_examples=10, deadline=None)
def test_nn_bce_loss_functions(input_data):
    run_test("test_nn_bce_loss", input_data, api_functions)

if __name__ == "__main__":
    test_nn_bce_loss_functions()
    finalize_results("test_nn_bce_loss")
