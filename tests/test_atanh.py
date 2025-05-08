from hypothesis import given, settings
from inputs.input_generator import generate_atanh_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_atanh
from functions.tf_functions import tf_atanh
from functions.jax_functions import jax_atanh
from functions.chainer_functions import chainer_atanh
from functions.keras_functions import keras_atanh

api_functions = {
    "pytorch_atanh": torch_atanh,
    "tensorflow_atanh": tf_atanh,
    "jax_atanh": jax_atanh,
    "chainer_atanh": chainer_atanh,
    "keras_atanh": keras_atanh,
}

@given(input_data=generate_atanh_inputs())
@settings(max_examples=100, deadline=None)
def test_atanh_functions(input_data):
    run_test("test_atanh", input_data, api_functions)

if __name__ == "__main__":
    test_atanh_functions()
    finalize_results("test_atanh")
