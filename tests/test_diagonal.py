from hypothesis import given, settings
from inputs.input_generator import generate_diagonal_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_diagonal
from functions.tf_functions import tf_diagonal
from functions.chainer_functions import chainer_diagonal

api_functions = {
    "pytorch_diagonal": torch_diagonal,
    "tensorflow_diagonal": tf_diagonal,
    "chainer_diagonal": chainer_diagonal,
}

@given(input_data=generate_diagonal_inputs())
@settings(max_examples=100, deadline=None)
def test_diagonal_functions(input_data):
    run_test("test_diagonal", input_data, api_functions)

if __name__ == "__main__":
    test_diagonal_functions()
    finalize_results("test_diagonal")
