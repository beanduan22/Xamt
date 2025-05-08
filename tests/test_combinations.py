from hypothesis import given, settings
from inputs.input_generator import generate_combinations_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_combinations

api_functions = {
    "pytorch_combinations": torch_combinations,
}

@given(input_data=generate_combinations_inputs())
@settings(max_examples=100, deadline=None)
def test_combinations_functions(input_data):
    run_test("test_combinations", input_data, api_functions)

if __name__ == "__main__":
    test_combinations_functions()
    finalize_results("test_combinations")
