from hypothesis import given, settings
from inputs.input_generator import generate_argsort_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_argsort
from functions.tf_functions import tf_argsort
from functions.jax_functions import jax_argsort

api_functions = {
    "pytorch_argsort": torch_argsort,
    "tensorflow_argsort": tf_argsort,
    "jax_argsort": jax_argsort,
}

@given(input_data=generate_argsort_inputs())
@settings(max_examples=100, deadline=None)
def test_argsort_functions(input_data):
    run_test("test_argsort", input_data, api_functions)

if __name__ == "__main__":
    test_argsort_functions()
    finalize_results("test_argsort")