from hypothesis import given, settings
from inputs.input_generator import generate_asinh_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_asinh
from functions.tf_functions import tf_asinh
from functions.jax_functions import jax_asinh

api_functions = {
    "pytorch_asinh": torch_asinh,
    "tensorflow_asinh": tf_asinh,
    "jax_asinh": jax_asinh,
}

@given(input_data=generate_asinh_inputs())
@settings(max_examples=100, deadline=None)
def test_asinh_functions(input_data):
    run_test("test_asinh", input_data, api_functions)

if __name__ == "__main__":
    test_asinh_functions()
    finalize_results("test_asinh")
