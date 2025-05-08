from hypothesis import given, settings
from inputs.input_generator import generate_bartlett_window_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bartlett_window
from functions.tf_functions import tf_bartlett_window
from functions.jax_functions import jax_bartlett_window
from functions.chainer_functions import chainer_bartlett_window
from functions.keras_functions import keras_bartlett_window

api_functions = {
    "pytorch_bartlett_window": torch_bartlett_window,
    "tensorflow_bartlett_window": tf_bartlett_window,
    "jax_bartlett_window": jax_bartlett_window,
    "chainer_bartlett_window": chainer_bartlett_window,
    "keras_bartlett_window": keras_bartlett_window,
}

@given(input_data=generate_bartlett_window_inputs())
@settings(max_examples=100, deadline=None)
def test_bartlett_window_functions(input_data):
    run_test("test_bartlett_window", input_data, api_functions)

if __name__ == "__main__":
    test_bartlett_window_functions()
    finalize_results("test_bartlett_window")
