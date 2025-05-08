from hypothesis import given, settings
from inputs.input_generator import generate_diff_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_diff
from functions.tf_functions import tf_diff

api_functions = {
    "pytorch_diff": torch_diff,
    "tensorflow_diff": tf_diff,
}

@given(input_data=generate_diff_inputs())
@settings(max_examples=100, deadline=None)
def test_diff_functions(input_data):
    run_test("test_diff", input_data, api_functions)

if __name__ == "__main__":
    test_diff_functions()
    finalize_results("test_diff")
