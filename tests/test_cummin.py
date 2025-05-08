from hypothesis import given, settings
from inputs.input_generator import generate_cummin_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cummin
from functions.tf_functions import tf_cummin
from functions.chainer_functions import chainer_cummin

api_functions = {
    "pytorch_cummin": torch_cummin,
    "tensorflow_cummin": tf_cummin,
    "chainer_cummin": chainer_cummin,
}

@given(input_data=generate_cummin_inputs())
@settings(max_examples=100, deadline=None)
def test_cummin_functions(input_data):
    run_test("test_cummin", input_data, api_functions)

if __name__ == "__main__":
    test_cummin_functions()
    finalize_results("test_cummin")
