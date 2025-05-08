from hypothesis import given, settings
from inputs.input_generator import generate_clamp_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_clamp
from functions.tf_functions import tf_clamp
from functions.chainer_functions import chainer_clamp
from functions.keras_functions import keras_clamp

api_functions = {
    "pytorch_clamp": torch_clamp,
    "tensorflow_clamp": tf_clamp,
    "chainer_clamp": chainer_clamp,
    "keras_clamp": keras_clamp,
}

@given(input_data=generate_clamp_inputs())
@settings(max_examples=100, deadline=None)
def test_clamp_functions(input_data):
    run_test("test_clamp", input_data, api_functions)

if __name__ == "__main__":
    test_clamp_functions()
    finalize_results("test_clamp")
