from hypothesis import given, settings
from inputs.input_generator import generate_clip_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_clip
from functions.tf_functions import tf_clip
from functions.chainer_functions import chainer_clip
from functions.keras_functions import keras_clip

api_functions = {
    "pytorch_clip": torch_clip,
    "tensorflow_clip": tf_clip,
    "chainer_clip": chainer_clip,
    "keras_clip": keras_clip,
}

@given(input_data=generate_clip_inputs())
@settings(max_examples=100, deadline=None)
def test_clip_functions(input_data):
    run_test("test_clip", input_data, api_functions)

if __name__ == "__main__":
    test_clip_functions()
    finalize_results("test_clip")
