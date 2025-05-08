from hypothesis import given, settings
from inputs.input_generator import generate_as_tensor_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_as_tensor
from functions.tf_functions import tf_constant

api_functions = {
    "pytorch_as_tensor": torch_as_tensor,
    "tensorflow_constant": tf_constant,
}

@given(input_data=generate_as_tensor_inputs())
@settings(max_examples=100, deadline=None)
def test_as_tensor_functions(input_data):
    run_test("test_as_tensor", input_data, api_functions)

if __name__ == "__main__":
    test_as_tensor_functions()
    finalize_results("test_as_tensor")
