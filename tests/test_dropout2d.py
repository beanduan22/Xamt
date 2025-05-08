from hypothesis import given, settings
from inputs.input_generator import generate_dropout2d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_dropout2d
from functions.tf_functions import tf_dropout2d
from functions.keras_functions import keras_dropout2d
from functions.chainer_functions import chainer_dropout2d
from functions.jax_functions import jax_dropout2d

api_functions = {
    "pytorch_dropout2d": torch_dropout2d,
    "tensorflow_dropout2d": tf_dropout2d,
    "keras_dropout2d": keras_dropout2d,
    "chainer_dropout2d": chainer_dropout2d,
    "jax_dropout2d": jax_dropout2d,
}

@given(input_data=generate_dropout2d_inputs())
@settings(max_examples=100, deadline=None)
def test_dropout2d_functions(input_data):
    run_test("test_dropout2d", input_data, api_functions)

if __name__ == "__main__":
    test_dropout2d_functions()
    finalize_results("test_dropout2d")
