from hypothesis import given, settings
from inputs.input_generator import generate_any_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_any
from functions.tf_functions import tf_any
from functions.jax_functions import jax_any
from functions.keras_functions import keras_any

api_functions = {
    "pytorch_any": torch_any,
    "tensorflow_any": tf_any,
    "keras_any": keras_any,
    "jax_any": jax_any,
}

@given(input_data=generate_any_inputs())
@settings(max_examples=100, deadline=None)
def test_any_functions(input_data):
    run_test("test_any", input_data, api_functions)

if __name__ == "__main__":
    test_any_functions()
    finalize_results("test_any")
