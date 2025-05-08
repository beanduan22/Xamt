from hypothesis import given, settings
from inputs.input_generator import generate_broadcast_to_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_broadcast_to
from functions.tf_functions import tf_broadcast_to
from functions.chainer_functions import chainer_broadcast_to

api_functions = {
    "pytorch_broadcast_to": torch_broadcast_to,
    "tensorflow_broadcast_to": tf_broadcast_to,
    "chainer_broadcast_to": chainer_broadcast_to,
}

@given(input_data=generate_broadcast_to_inputs())
@settings(max_examples=100, deadline=None)
def test_broadcast_to_functions(input_data):
    run_test("test_broadcast_to", input_data, api_functions)

if __name__ == "__main__":
    test_broadcast_to_functions()
    finalize_results("test_broadcast_to")
