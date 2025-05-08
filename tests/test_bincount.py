from hypothesis import given, settings
from inputs.input_generator import generate_bincount_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bincount
from functions.tf_functions import tf_bincount
from functions.chainer_functions import chainer_bincount

api_functions = {
    "pytorch_bincount": torch_bincount,
    "tensorflow_bincount": tf_bincount,
    "chainer_bincount": chainer_bincount,
}

@given(input_data=generate_bincount_inputs())
@settings(max_examples=100, deadline=None)
def test_bincount_functions(input_data):
    run_test("test_bincount", input_data, api_functions)

if __name__ == "__main__":
    test_bincount_functions()
    finalize_results("test_bincount")
