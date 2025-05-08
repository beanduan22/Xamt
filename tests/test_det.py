from hypothesis import given, settings
from inputs.input_generator import generate_det_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_det
from functions.tf_functions import tf_det
from functions.chainer_functions import chainer_det

api_functions = {
    "pytorch_det": torch_det,
    "tensorflow_det": tf_det,
    "chainer_det": chainer_det,
}

@given(input_data=generate_det_inputs())
@settings(max_examples=100, deadline=None)
def test_det_functions(input_data):
    run_test("test_det", input_data, api_functions)

if __name__ == "__main__":
    test_det_functions()
    finalize_results("test_det")
