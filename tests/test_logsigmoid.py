from hypothesis import given, settings
from inputs.input_generator import generate_logsigmoid_input
from functions.torch_functions import nn_torch_logsigmoid
from utilities.helpers import run_test, finalize_results

@given(input_data=generate_logsigmoid_input())
@settings(max_examples=100, deadline=None)
def test_logsigmoid_functions(input_data):
    run_test("test_logsigmoid", input_data, nn_torch_logsigmoid)

if __name__ == "__main__":
    test_logsigmoid_functions()
    finalize_results("test_logsigmoid")
