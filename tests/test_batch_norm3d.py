from hypothesis import given, settings
from inputs.input_generator import generate_batch_norm_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_batch_norm3d
from functions.tf_functions import tf_batch_norm
from functions.chainer_functions import chainer_batch_norm

api_functions = {
    "pytorch_batch_norm3d": torch_batch_norm3d,
    "tensorflow_batch_norm": tf_batch_norm,
    "chainer_batch_norm": chainer_batch_norm,
}

@given(input_data=generate_batch_norm_inputs())
@settings(max_examples=100, deadline=None)
def test_batch_norm3d_functions(input_data):
    run_test("test_batch_norm3d", input_data, api_functions)

if __name__ == "__main__":
    test_batch_norm3d_functions()
    finalize_results("test_batch_norm3d")
