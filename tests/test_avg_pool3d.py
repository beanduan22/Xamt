from hypothesis import given, settings
from inputs.input_generator import generate_avg_pool3d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_avg_pool3d
from functions.tf_functions import tf_avg_pool3d
from functions.chainer_functions import chainer_avg_pool3d

api_functions = {
    "pytorch_avg_pool3d": torch_avg_pool3d,
    "tensorflow_avg_pool3d": tf_avg_pool3d,
    "chainer_avg_pool3d": chainer_avg_pool3d,
}

@given(input_data=generate_avg_pool3d_inputs())
@settings(max_examples=100, deadline=None)
def test_avg_pool3d_functions(input_data):
    run_test("test_avg_pool3d", input_data, api_functions)

if __name__ == "__main__":
    test_avg_pool3d_functions()
    finalize_results("test_avg_pool3d")
