from hypothesis import given, settings
from inputs.input_generator import generate_nn_flatten_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_nn_flatten
from functions.tf_functions import tf_nn_flatten
from functions.keras_functions import keras_nn_flatten
from functions.mxnet_functions import mxnet_nn_flatten
from functions.jax_functions import jax_nn_flatten

api_functions = {
    "pytorch_nn_flatten": torch_nn_flatten,
    "tensorflow_nn_flatten": tf_nn_flatten,
    "keras_nn_flatten": keras_nn_flatten,
    "mxnet_nn_flatten": mxnet_nn_flatten,
    "jax_nn_flatten": jax_nn_flatten,
}

@given(input_data=generate_nn_flatten_input())
@settings(max_examples=100, deadline=None)
def test_nn_flatten(input_data):
    run_test("test_nn_flatten", input_data, api_functions)

if __name__ == "__main__":
    test_nn_flatten()
    finalize_results("test_nn_flatten")
