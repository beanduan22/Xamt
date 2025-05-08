from hypothesis import given, settings
from inputs.input_generator import generate_bucketize_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bucketize
from functions.tf_functions import tf_bucketize
from functions.chainer_functions import chainer_bucketize

api_functions = {
    "pytorch_bucketize": torch_bucketize,
    "tensorflow_bucketize": tf_bucketize,
    "chainer_bucketize": chainer_bucketize,
}

@given(input_data=generate_bucketize_inputs())
@settings(max_examples=100, deadline=None)
def test_bucketize_functions(input_data):
    run_test("test_bucketize", input_data, api_functions)

if __name__ == "__main__":
    test_bucketize_functions()
    finalize_results("test_bucketize")
