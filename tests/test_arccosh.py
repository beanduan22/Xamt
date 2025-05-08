from hypothesis import given, settings
from inputs.input_generator import generate_arccosh_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_arccosh
from functions.tf_functions import tf_arccosh
from functions.jax_functions import jnp_arccosh

api_functions = {
    "pytorch_arccosh": torch_arccosh,
    "tensorflow_arccosh": tf_arccosh,
    "jax_arccosh": jnp_arccosh,
}

@given(input_data=generate_arccosh_inputs())
@settings(max_examples=100, deadline=None)
def test_arccosh_functions(input_data):
    run_test("test_arccosh", input_data, api_functions)

if __name__ == "__main__":
    test_arccosh_functions()
    finalize_results("test_arccosh")
