from hypothesis import given, settings
from inputs.input_generator import generate_arange_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_arange
from functions.tf_functions import tf_arange
from functions.keras_functions import keras_arange
from functions.jax_functions import jax_arange

api_functions = {
    "pytorch_arange": torch_arange,
    "tensorflow_arange": tf_arange,
    "keras_arange": keras_arange,
    "jax_arange": jax_arange,
}

@given(input_data=generate_arange_inputs())
@settings(max_examples=100, deadline=None)
def test_arange_functions(input_data):
    run_test("test_arange", input_data, api_functions)

if __name__ == "__main__":
    test_arange_functions()
    finalize_results("test_arange")
