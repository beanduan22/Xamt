from hypothesis import given, settings
from inputs.input_generator import generate_linear_input
from functions.torch_functions import nn_torch_linear
from functions.tf_functions import tf_linear
from functions.keras_functions import keras_linear
from functions.chainer_functions import chainer_linear
from functions.jax_functions import jax_linear
from utilities.helpers import run_test, finalize_results

api_functions = {
    "pytorch_linear": nn_torch_linear,
    "tensorflow_linear": tf_linear,
    "keras_linear": keras_linear,
    "chainer_linear": chainer_linear,
    "jax_linear": jax_linear,
}

@given(input_data=generate_linear_input())
@settings(max_examples=100, deadline=None)
def test_linear_functions(input_data):
    run_test("test_linear", input_data, api_functions)

if __name__ == "__main__":
    test_linear_functions()
    finalize_results("test_linear")
