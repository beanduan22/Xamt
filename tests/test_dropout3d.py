from hypothesis import given, settings
from inputs.input_generator import generate_dropout3d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_dropout3d

api_functions = {
    "pytorch_dropout3d": torch_dropout3d,
}

@given(input_data=generate_dropout3d_inputs())
@settings(max_examples=100, deadline=None)
def test_dropout3d_functions(input_data):
    run_test("test_dropout3d", input_data, api_functions)

if __name__ == "__main__":
    test_dropout3d_functions()
    finalize_results("test_dropout3d")
