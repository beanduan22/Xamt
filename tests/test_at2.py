from hypothesis import given, settings
from inputs.input_generator import generate_atleast_2d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_atleast_2d
from functions.tf_functions import tf_atleast_2d
from functions.jax_functions import jax_atleast_2d
from functions.chainer_functions import chainer_atleast_2d
from functions.keras_functions import keras_atleast_2d

api_functions = {
    "pytorch_atleast_2d": torch_atleast_2d,
    "tensorflow_atleast_2d": tf_atleast_2d,
    "jax_atleast_2d": jax_atleast_2d,
    "chainer_atleast_2d": chainer_atleast_2d,
    "keras_atleast_2d": keras_atleast_2d,
}

@given(input_data=generate_atleast_2d_inputs())
@settings(max_examples=100, deadline=None)
def test_atleast_2d_functions(input_data):
    run_test("test_atleast_2d", input_data, api_functions)

if __name__ == "__main__":
    test_atleast_2d_functions()
    finalize_results("test_atleast_2d")
