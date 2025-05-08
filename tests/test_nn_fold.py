from hypothesis import given, settings
from inputs.input_generator import generate_nn_fold_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_nn_fold
from functions.mxnet_functions import mxnet_nn_fold

api_functions = {
    "pytorch_nn_fold": torch_nn_fold,
    "mxnet_nn_fold": mxnet_nn_fold,
}

@given(input_data=generate_nn_fold_input())
@settings(max_examples=100, deadline=None)
def test_nn_fold(input_data):
    run_test("test_nn_fold", input_data, api_functions)

if __name__ == "__main__":
    test_nn_fold()
    finalize_results("test_nn_fold")
