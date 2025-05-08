from hypothesis import given, settings
from inputs.input_generator import generate_arctan_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_arctan
from functions.tf_functions import tf_arctan
from functions.jax_functions import jax_arctan

api_functions = {
    "pytorch_arctan": torch_arctan,
    "tensorflow_arctan": tf_arctan,
    "jax_arctan": jax_arctan,
}

@given(input_data=generate_arctan_inputs())
@settings(max_examples=100, deadline=None)
def test_arctan_functions(input_data):
    run_test("test_arctan", input_data, api_functions)

if __name__ == "__main__":
    test_arctan_functions()
    finalize_results("test_arctan")
