from hypothesis import given, settings
from inputs.input_generator import generate_cartesian_prod_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cartesian_prod

api_functions = {
    "pytorch_cartesian_prod": torch_cartesian_prod,
}

@given(input_data=generate_cartesian_prod_inputs())
@settings(max_examples=100, deadline=None)
def test_cartesian_prod_functions(input_data):
    run_test("test_cartesian_prod", input_data, api_functions)

if __name__ == "__main__":
    test_cartesian_prod_functions()
    finalize_results("test_cartesian_prod")
