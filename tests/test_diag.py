from hypothesis import given, settings
from inputs.input_generator import generate_diag_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_diag
from functions.tf_functions import tf_diag
from functions.chainer_functions import chainer_diag

api_functions = {
    "pytorch_diag": torch_diag,
    "tensorflow_diag": tf_diag,
    "chainer_diag": chainer_diag,
}

@given(input_data=generate_diag_inputs())
@settings(max_examples=100, deadline=None)
def test_diag_functions(input_data):
    run_test("test_diag", input_data, api_functions)

if __name__ == "__main__":
    test_diag_functions()
    finalize_results("test_diag")
