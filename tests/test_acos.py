from hypothesis import given, settings
from inputs.input_generator import acos_input_strategy
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_acos
from functions.tf_functions import tf_acos
from functions.chainer_functions import chainer_acos
from functions.keras_functions import keras_acos
from functions.jax_functions import jax_acos

api_functions = {
    "pytorch_acos": torch_acos,
    "tensorflow_acos": tf_acos,
    "chainer_acos": chainer_acos,
    "keras_acos": keras_acos,
    "jax_acos": jax_acos,
}

@given(input_data=acos_input_strategy())
@settings(max_examples=1000, deadline=None)
def test_acos_functions(input_data):
    run_test("test_acos", input_data, api_functions)

if __name__ == "__main__":
    test_acos_functions()
    finalize_results("test_acos")
