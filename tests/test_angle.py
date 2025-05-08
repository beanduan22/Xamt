from hypothesis import given, settings
from inputs.input_generator import generate_angle_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_angle
from functions.jax_functions import jax_angle

api_functions = {
    "pytorch_angle": torch_angle,
    "jax_angle": jax_angle,
}

@given(input_data=generate_angle_inputs())
@settings(max_examples=100, deadline=None)
def test_angle_functions(input_data):
    run_test("test_angle", input_data, api_functions)

if __name__ == "__main__":
    test_angle_functions()
    finalize_results("test_angle")
