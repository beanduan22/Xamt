from hypothesis import given, settings
from inputs.input_generator import generate_atan_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_atan
from functions.tf_functions import tf_atan
from functions.jax_functions import jax_atan
from functions.chainer_functions import chainer_atan
from functions.keras_functions import keras_atan

api_functions = {
    "pytorch_atan": torch_atan,
    "tensorflow_atan": tf_atan,
    "jax_atan": jax_atan,
    "chainer_atan": chainer_atan,
    "keras_atan": keras_atan,
}

@given(input_data=generate_atan_inputs())
@settings(max_examples=100, deadline=None)
def test_atan_functions(input_data):
    run_test("test_atan", input_data, api_functions)

if __name__ == "__main__":
    test_atan_functions()
    finalize_results("test_atan")
