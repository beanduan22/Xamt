from hypothesis import given, settings
from inputs.input_generator import generate_atleast_1d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_atleast_1d
from functions.tf_functions import tf_atleast_1d
from functions.jax_functions import jax_atleast_1d
from functions.chainer_functions import chainer_atleast_1d
from functions.keras_functions import keras_atleast_1d

api_functions = {
    "pytorch_atleast_1d": torch_atleast_1d,
    "tensorflow_atleast_1d": tf_atleast_1d,
    "jax_atleast_1d": jax_atleast_1d,
    "chainer_atleast_1d": chainer_atleast_1d,
    "keras_atleast_1d": keras_atleast_1d,
}

@given(input_data=generate_atleast_1d_inputs())
@settings(max_examples=100, deadline=None)
def test_atleast_1d_functions(input_data):
    run_test("test_atleast_1d", input_data, api_functions)

if __name__ == "__main__":
    test_atleast_1d_functions()
    finalize_results("test_atleast_1d")