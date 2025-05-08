from hypothesis import given, settings
from inputs.input_generator import generate_alpha_dropout_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_alpha_dropout
from functions.tf_functions import tf_alpha_dropout

api_functions = {
    "pytorch_alpha_dropout": torch_alpha_dropout,
    "tensorflow_alpha_dropout": tf_alpha_dropout,
}

@given(input_data=generate_alpha_dropout_inputs())
@settings(max_examples=100, deadline=None)
def test_alpha_dropout_functions(input_data):
    run_test("test_alpha_dropout", input_data, api_functions)

if __name__ == "__main__":
    test_alpha_dropout_functions()
    finalize_results("test_alpha_dropout")
