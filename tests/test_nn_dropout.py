from hypothesis import given, settings
from inputs.input_generator import generate_nn_dropout_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_dropout
from functions.tf_functions import nn_tf_dropout
from functions.chainer_functions import nn_chainer_dropout
from functions.keras_functions import nn_keras_dropout
from functions.jax_functions import nn_jax_dropout

api_functions = {
    "pytorch_dropout": nn_torch_dropout,
    "tensorflow_dropout": nn_tf_dropout,
    "chainer_dropout": nn_chainer_dropout,
    "keras_dropout": nn_keras_dropout,
    "jax_dropout": nn_jax_dropout,
}

@given(input_data=generate_nn_dropout_input())
@settings(max_examples=100, deadline=None)
def test_dropout_functions(input_data):
    run_test("test_dropout", input_data, api_functions)

if __name__ == "__main__":
    test_dropout_functions()
    finalize_results("test_dropout")
