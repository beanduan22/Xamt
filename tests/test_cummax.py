from hypothesis import given, settings
from inputs.input_generator import generate_cummax_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cummax
from functions.tf_functions import tf_cummax
from functions.chainer_functions import chainer_cummax

api_functions = {
    "pytorch_cummax": torch_cummax,
    "tensorflow_cummax": tf_cummax,
    "chainer_cummax": chainer_cummax,
}

@given(input_data=generate_cummax_inputs())
@settings(max_examples=100, deadline=None)
def test_cummax_functions(input_data):
    run_test("test_cummax", input_data, api_functions)

if __name__ == "__main__":
    test_cummax_functions()
    finalize_results("test_cummax")
