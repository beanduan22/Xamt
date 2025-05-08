from hypothesis import given, settings
from inputs.input_generator import generate_blackman_window_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_blackman_window
from functions.tf_functions import tf_blackman_window
from functions.jax_functions import jax_blackman_window

api_functions = {
    "pytorch_blackman_window": torch_blackman_window,
    "tensorflow_blackman_window": tf_blackman_window,
    "jax_blackman_window": jax_blackman_window,
}

@given(input_data=generate_blackman_window_inputs())
@settings(max_examples=100, deadline=None)
def test_blackman_window_functions(input_data):
    run_test("test_blackman_window", input_data, api_functions)

if __name__ == "__main__":
    test_blackman_window_functions()
    finalize_results("test_blackman_window")
