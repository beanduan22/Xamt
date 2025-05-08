from hypothesis import given, settings
from inputs.input_generator import generate_arctanh_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_arctanh
from functions.tf_functions import tf_arctanh
from functions.jax_functions import jax_arctanh

api_functions = {
    "pytorch_arctanh": torch_arctanh,
    "tensorflow_arctanh": tf_arctanh,
    "jax_arctanh": jax_arctanh,
}

@given(input_data=generate_arctanh_inputs())
@settings(max_examples=100, deadline=None)
def test_arctanh_functions(input_data):
    run_test("test_arctanh", input_data, api_functions)

if __name__ == "__main__":
    test_arctanh_functions()
    finalize_results("test_arctanh")
