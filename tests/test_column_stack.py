from hypothesis import given, settings
from inputs.input_generator import generate_column_stack_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_column_stack
from functions.tf_functions import tf_column_stack
from functions.chainer_functions import chainer_column_stack

api_functions = {
    "pytorch_column_stack": torch_column_stack,
    "tensorflow_column_stack": tf_column_stack,
    "chainer_column_stack": chainer_column_stack,
}

@given(input_data=generate_column_stack_inputs())
@settings(max_examples=100, deadline=None)
def test_column_stack_functions(input_data):
    run_test("test_column_stack", input_data, api_functions)

if __name__ == "__main__":
    test_column_stack_functions()
    finalize_results("test_column_stack")
