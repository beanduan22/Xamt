from hypothesis import given, settings
from inputs.input_generator import generate_cholesky_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cholesky
from functions.tf_functions import tf_cholesky
from functions.chainer_functions import chainer_cholesky

api_functions = {
    "pytorch_cholesky": torch_cholesky,
    "tensorflow_cholesky": tf_cholesky,
    "chainer_cholesky": chainer_cholesky,
}

@given(input_data=generate_cholesky_inputs())
@settings(max_examples=100, deadline=None)
def test_cholesky_functions(input_data):
    run_test("test_cholesky", input_data, api_functions)

if __name__ == "__main__":
    test_cholesky_functions()
    finalize_results("test_cholesky")
