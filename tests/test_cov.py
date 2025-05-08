from hypothesis import given, settings
from inputs.input_generator import generate_cov_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cov
from functions.tf_functions import tf_cov

api_functions = {
    "pytorch_cov": torch_cov,
    "tensorflow_cov": tf_cov,
}

@given(input_data=generate_cov_inputs())
@settings(max_examples=100, deadline=None)
def test_cov_functions(input_data):
    run_test("test_cov", input_data, api_functions)

if __name__ == "__main__":
    test_cov_functions()
    finalize_results("test_cov")
