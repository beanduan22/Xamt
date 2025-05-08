from hypothesis import given, settings
from inputs.input_generator import generate_diagflat_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_diagflat
from functions.tf_functions import tf_diagflat

api_functions = {
    "pytorch_diagflat": torch_diagflat,
    "tensorflow_diagflat": tf_diagflat,
}

@given(input_data=generate_diagflat_inputs())
@settings(max_examples=100, deadline=None)
def test_diagflat_functions(input_data):
    run_test("test_diagflat", input_data, api_functions)

if __name__ == "__main__":
    test_diagflat_functions()
    finalize_results("test_diagflat")
