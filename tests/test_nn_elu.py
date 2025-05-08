from hypothesis import given, settings
from inputs.input_generator import generate_nn_elu_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_elu
from functions.tf_functions import nn_tf_elu
from functions.chainer_functions import nn_chainer_elu
from functions.keras_functions import nn_keras_elu
from functions.jax_functions import nn_jax_elu

api_functions = {
    "pytorch_elu": nn_torch_elu,
    "tensorflow_elu": nn_tf_elu,
    "chainer_elu": nn_chainer_elu,
    "keras_elu": nn_keras_elu,
    "jax_elu": nn_jax_elu,
}

@given(input_data=generate_nn_elu_input())
@settings(max_examples=100, deadline=None)
def test_elu_functions(input_data):
    run_test("test_elu", input_data, api_functions)

if __name__ == "__main__":
    test_elu_functions()
    finalize_results("test_elu")
