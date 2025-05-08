from hypothesis import given, settings
from inputs.input_generator import generate_cdist_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cdist
from functions.tf_functions import tf_cdist
from functions.chainer_functions import chainer_cdist

api_functions = {
    "pytorch_cdist": torch_cdist,
    "tensorflow_cdist": tf_cdist,
    "chainer_cdist": chainer_cdist,
}

@given(input_data=generate_cdist_inputs())
@settings(max_examples=100, deadline=None)
def test_cdist_functions(input_data):
    run_test("test_cdist", input_data, api_functions)

if __name__ == "__main__":
    test_cdist_functions()
    finalize_results("test_cdist")
