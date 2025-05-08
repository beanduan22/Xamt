from hypothesis import given, settings
from inputs.input_generator import generate_broadcast_shapes_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_broadcast_shapes
from functions.tf_functions import tf_broadcast_shapes
from functions.chainer_functions import chainer_broadcast_shapes

api_functions = {
    "pytorch_broadcast_shapes": torch_broadcast_shapes,
    "tensorflow_broadcast_shapes": tf_broadcast_shapes,
    "chainer_broadcast_shapes": chainer_broadcast_shapes,
}

@given(input_data=generate_broadcast_shapes_inputs())
@settings(max_examples=100, deadline=None)
def test_broadcast_shapes_functions(input_data):
    run_test("test_broadcast_shapes", input_data, api_functions)

if __name__ == "__main__":
    test_broadcast_shapes_functions()
    finalize_results("test_broadcast_shapes")
