from hypothesis import given, settings
from inputs.input_generator import generate_complex_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_complex
from functions.tf_functions import tf_complex

api_functions = {
    "pytorch_complex": torch_complex,
    "tensorflow_complex": tf_complex,
}

@given(input_data=generate_complex_inputs())
@settings(max_examples=100, deadline=None)
def test_complex_functions(input_data):
    run_test("test_complex", input_data, api_functions)

if __name__ == "__main__":
    test_complex_functions()
    finalize_results("test_complex")
