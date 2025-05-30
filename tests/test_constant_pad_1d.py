from hypothesis import given, settings
from inputs.input_generator import generate_constant_pad_1d_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_constant_pad_1d
from functions.tf_functions import tf_constant_pad_1d
from functions.chainer_functions import chainer_constant_pad_1d

api_functions = {
    "pytorch_constant_pad_1d": torch_constant_pad_1d,
    "tensorflow_constant_pad_1d": tf_constant_pad_1d,
    "chainer_constant_pad_1d": chainer_constant_pad_1d,
}

@given(input_data=generate_constant_pad_1d_inputs())
@settings(max_examples=100, deadline=None)
def test_constant_pad_1d_functions(input_data):
    run_test("test_constant_pad_1d", input_data, api_functions)

if __name__ == "__main__":
    test_constant_pad_1d_functions()
    finalize_results("test_constant_pad_1d")
