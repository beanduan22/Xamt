from hypothesis import given, settings
from inputs.input_generator import generate_cosh_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cosh
from functions.tf_functions import tf_cosh
from functions.keras_functions import keras_cosh
from functions.chainer_functions import chainer_cosh
from functions.jax_functions import jax_cosh

api_functions = {
    "pytorch_cosh": torch_cosh,
    "tensorflow_cosh": tf_cosh,
    "keras_cosh": keras_cosh,
    "chainer_cosh": chainer_cosh,
    "jax_cosh": jax_cosh,
}

@given(input_data=generate_cosh_inputs())
@settings(max_examples=100, deadline=None)
def test_cosh_functions(input_data):
    run_test("test_cosh", input_data, api_functions)

if __name__ == "__main__":
    test_cosh_functions()
    finalize_results("test_cosh")
