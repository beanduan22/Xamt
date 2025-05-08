from hypothesis import given, settings
from inputs.input_generator import generate_corrcoef_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_corrcoef
from functions.tf_functions import tf_corrcoef

api_functions = {
    "pytorch_corrcoef": torch_corrcoef,
    "tensorflow_corrcoef": tf_corrcoef,
}

@given(input_data=generate_corrcoef_inputs())
@settings(max_examples=100, deadline=None)
def test_corrcoef_functions(input_data):
    run_test("test_corrcoef", input_data, api_functions)

if __name__ == "__main__":
    test_corrcoef_functions()
    finalize_results("test_corrcoef")
