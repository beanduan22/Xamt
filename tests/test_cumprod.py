from hypothesis import given, settings
from inputs.input_generator import generate_cumprod_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cumprod
from functions.tf_functions import tf_cumprod
from functions.chainer_functions import chainer_cumprod
from functions.keras_functions import keras_cumprod

api_functions = {
    "pytorch_cumprod": torch_cumprod,
    "tensorflow_cumprod": tf_cumprod,
    "chainer_cumprod": chainer_cumprod,
    "keras_cumprod": keras_cumprod,
}

@given(input_data=generate_cumprod_inputs())
@settings(max_examples=100, deadline=None)
def test_cumprod_functions(input_data):
    run_test("test_cumprod", input_data, api_functions)

if __name__ == "__main__":
    test_cumprod_functions()
    finalize_results("test_cumprod")
