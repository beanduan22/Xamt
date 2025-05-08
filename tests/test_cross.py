from hypothesis import given, settings
from inputs.input_generator import generate_cross_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cross
from functions.tf_functions import tf_cross
from functions.chainer_functions import chainer_cross

api_functions = {
    "pytorch_cross": torch_cross,
    "tensorflow_cross": tf_cross,
    "chainer_cross": chainer_cross,
}

@given(input_data=generate_cross_inputs())
@settings(max_examples=100, deadline=None)
def test_cross_functions(input_data):
    run_test("test_cross", input_data, api_functions)

if __name__ == "__main__":
    test_cross_functions()
    finalize_results("test_cross")
