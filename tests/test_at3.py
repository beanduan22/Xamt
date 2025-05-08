from hypothesis import given, settings
from inputs.input_generator import generate_atleast_3d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_atleast_3d
from functions.tf_functions import tf_atleast_3d
from functions.jax_functions import jax_atleast_3d
from functions.chainer_functions import chainer_atleast_3d
from functions.keras_functions import keras_atleast_3d

api_functions = {
    "pytorch_atleast_3d": torch_atleast_3d,
    "tensorflow_atleast_3d": tf_atleast_3d,
    "jax_atleast_3d": jax_atleast_3d,
    "chainer_atleast_3d": chainer_atleast_3d,
    "keras_atleast_3d": keras_atleast_3d,
}

@given(input_data=generate_atleast_3d_inputs())
@settings(max_examples=100, deadline=None)
def test_atleast_3d_functions(input_data):
    run_test("test_atleast_3d", input_data, api_functions)

if __name__ == "__main__":
    test_atleast_3d_functions()
    finalize_results("test_atleast_3d")
