from hypothesis import given, settings
from inputs.input_generator import generate_cholesky_inverse_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cholesky_inverse
from functions.tf_functions import tf_cholesky_inverse
from functions.chainer_functions import chainer_cholesky_inverse

api_functions = {
    "pytorch_cholesky_inverse": torch_cholesky_inverse,
    "tensorflow_cholesky_inverse": tf_cholesky_inverse,
    "chainer_cholesky_inverse": chainer_cholesky_inverse,
}

@given(input_data=generate_cholesky_inverse_inputs())
@settings(max_examples=100, deadline=None)
def test_cholesky_inverse_functions(input_data):
    run_test("test_cholesky_inverse", input_data, api_functions)

if __name__ == "__main__":
    test_cholesky_inverse_functions()
    finalize_results("test_cholesky_inverse")
