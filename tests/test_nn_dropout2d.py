from hypothesis import given, settings
from inputs.input_generator import generate_nn_dropout2d_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_dropout2d
from functions.tf_functions import nn_tf_dropout2d
from functions.chainer_functions import nn_chainer_dropout2d
from functions.keras_functions import nn_keras_dropout2d
from functions.jax_functions import nn_jax_dropout2d

api_functions = {
    "pytorch_dropout2d": nn_torch_dropout2d,
    "tensorflow_dropout2d": nn_tf_dropout2d,
    "chainer_dropout2d": nn_chainer_dropout2d,
    "keras_dropout2d": nn_keras_dropout2d,
    "jax_dropout2d": nn_jax_dropout2d,
}

@given(input_data=generate_nn_dropout2d_input())
@settings(max_examples=100, deadline=None)
def test_dropout2d_functions(input_data):
    run_test("test_dropout2d", input_data, api_functions)

if __name__ == "__main__":
    test_dropout2d_functions()
    finalize_results("test_dropout2d")