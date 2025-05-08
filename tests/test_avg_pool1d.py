from hypothesis import given, settings
from inputs.input_generator import generate_avg_pool1d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_avg_pool1d
from functions.tf_functions import tf_avg_pool1d
from functions.chainer_functions import chainer_avg_pool1d

api_functions = {
    "pytorch_avg_pool1d": torch_avg_pool1d,
    "tensorflow_avg_pool1d": tf_avg_pool1d,
    "chainer_avg_pool1d": chainer_avg_pool1d,
}

@given(input_data=generate_avg_pool1d_inputs())
@settings(max_examples=100, deadline=None)
def test_avg_pool1d_functions(input_data):
    run_test("test_avg_pool1d", input_data, api_functions)

if __name__ == "__main__":
    test_avg_pool1d_functions()
    finalize_results("test_avg_pool1d")
