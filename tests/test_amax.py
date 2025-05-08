from hypothesis import given, settings
from inputs.input_generator import generate_amax_inputs
from utilities.helper1 import run_test, finalize_results
from functions.torch_functions import torch_amax
from functions.tf_functions import tf_amax
from functions.keras_functions import keras_amax
from functions.jax_functions import jax_amax

api_functions = {
    "pytorch_amax": torch_amax,
    "tensorflow_amax": tf_amax,
    "keras_amax": keras_amax,
    "jax_amax": jax_amax,
}

@given(input_data=generate_amax_inputs())
@settings(max_examples=100, deadline=None)
def test_amax_functions(input_data):
    run_test("test_amax", input_data, api_functions)

if __name__ == "__main__":
    test_amax_functions()
    finalize_results("test_amax")
