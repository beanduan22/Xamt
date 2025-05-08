from hypothesis import given, settings
from inputs.input_generator import generate_count_nonzero_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_count_nonzero
from functions.tf_functions import tf_count_nonzero
from functions.chainer_functions import chainer_count_nonzero

api_functions = {
    "pytorch_count_nonzero": torch_count_nonzero,
    "tensorflow_count_nonzero": tf_count_nonzero,
    "chainer_count_nonzero": chainer_count_nonzero,
}

@given(input_data=generate_count_nonzero_inputs())
@settings(max_examples=100, deadline=None)
def test_count_nonzero_functions(input_data):
    run_test("test_count_nonzero", input_data, api_functions)

if __name__ == "__main__":
    test_count_nonzero_functions()
    finalize_results("test_count_nonzero")
