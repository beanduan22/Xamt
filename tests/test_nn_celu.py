from hypothesis import given, strategies as st, settings
from inputs.input_generator import generate_nn_celu_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_celu
from functions.tf_functions import nn_tf_celu
from functions.keras_functions import nn_keras_celu
from functions.chainer_functions import nn_chainer_celu
from functions.jax_functions import nn_jax_celu

api_functions = {
    "pytorch_nn_celu": nn_torch_celu,
    "tensorflow_nn_celu": nn_tf_celu,
    "keras_nn_celu": nn_keras_celu,
    "chainer_nn_celu": nn_chainer_celu,
    "jax_nn_celu": nn_jax_celu,
}

@given(input_data=generate_nn_celu_input())
@settings(max_examples=10, deadline=None)
def test_nn_celu_functions(input_data):
    run_test("test_nn_celu", input_data, api_functions)

if __name__ == "__main__":
    test_nn_celu_functions()
    finalize_results("test_nn_celu")
