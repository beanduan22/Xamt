from hypothesis import given, settings
from inputs.input_generator import generate_ceil_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_ceil
from functions.tf_functions import tf_ceil
from functions.chainer_functions import chainer_ceil
from functions.keras_functions import keras_ceil

api_functions = {
    "pytorch_ceil": torch_ceil,
    "tensorflow_ceil": tf_ceil,
    "chainer_ceil": chainer_ceil,
    "keras_ceil": keras_ceil,
}

@given(input_data=generate_ceil_inputs())
@settings(max_examples=100, deadline=None)
def test_ceil_functions(input_data):
    run_test("test_ceil", input_data, api_functions)

if __name__ == "__main__":
    test_ceil_functions()
    finalize_results("test_ceil")
