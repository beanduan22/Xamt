from hypothesis import given, settings
from inputs.input_generator import generate_adaptive_max_pool1d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_adaptive_max_pool1d
from functions.tf_functions import tf_adaptive_max_pool1d
from functions.chainer_functions import chainer_adaptive_max_pool1d

api_functions = {
    "pytorch_adaptive_max_pool1d": torch_adaptive_max_pool1d,
    "tensorflow_adaptive_max_pool1d": tf_adaptive_max_pool1d,
    "chainer_adaptive_max_pool1d": chainer_adaptive_max_pool1d,
}

@given(input_data=generate_adaptive_max_pool1d_inputs())
@settings(max_examples=100, deadline=None)
def test_adaptive_max_pool1d_functions(input_data):
    run_test("test_adaptive_max_pool1d", input_data, api_functions)

if __name__ == "__main__":
    test_adaptive_max_pool1d_functions()
    finalize_results("test_adaptive_max_pool1d")
