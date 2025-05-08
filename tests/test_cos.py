from hypothesis import given, settings
from inputs.input_generator import generate_cos_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cos
from functions.tf_functions import tf_cos
from functions.keras_functions import keras_cos
from functions.chainer_functions import chainer_cos
from functions.jax_functions import jax_cos

api_functions = {
    "pytorch_cos": torch_cos,
    "tensorflow_cos": tf_cos,
    "keras_cos": keras_cos,
    "chainer_cos": chainer_cos,
    "jax_cos": jax_cos,
}

@given(input_data=generate_cos_inputs())
@settings(max_examples=100, deadline=None)
def test_cos_functions(input_data):
    run_test("test_cos", input_data, api_functions)

if __name__ == "__main__":
    test_cos_functions()
    finalize_results("test_cos")
