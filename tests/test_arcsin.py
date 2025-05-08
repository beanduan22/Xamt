from hypothesis import given, settings
from inputs.input_generator import generate_arcsin_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_arcsin
from functions.tf_functions import tf_arcsin
from functions.jax_functions import jax_arcsin

api_functions = {
    "pytorch_arcsin": torch_arcsin,
    "tensorflow_arcsin": tf_arcsin,
    "jax_arcsin": jax_arcsin,
}

@given(input_data=generate_arcsin_inputs())
@settings(max_examples=100, deadline=None)
def test_arcsin_functions(input_data):
    run_test("test_arcsin", input_data, api_functions)

if __name__ == "__main__":
    test_arcsin_functions()
    finalize_results("test_arcsin")
