from hypothesis import given, settings
from inputs.input_generator import generate_broadcast_tensors_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_broadcast_tensors
from functions.tf_functions import tf_broadcast_tensors
from functions.chainer_functions import chainer_broadcast_tensors

api_functions = {
    "pytorch_broadcast_tensors": torch_broadcast_tensors,
    "tensorflow_broadcast_tensors": tf_broadcast_tensors,
    "chainer_broadcast_tensors": chainer_broadcast_tensors,
}

@given(input_data=generate_broadcast_tensors_inputs())
@settings(max_examples=100, deadline=None)
def test_broadcast_tensors_functions(input_data):
    run_test("test_broadcast_tensors", input_data, api_functions)

if __name__ == "__main__":
    test_broadcast_tensors_functions()
    finalize_results("test_broadcast_tensors")
