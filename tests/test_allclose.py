from hypothesis import given, settings
from inputs.input_generator import generate_allclose_inputs
from utilities.helper1 import run_test, finalize_results
from functions.torch_functions import torch_allclose
from functions.tf_functions import tf_allclose
from functions.keras_functions import keras_allclose
from functions.jax_functions import jax_allclose

api_functions = {
    "pytorch_allclose": torch_allclose,
    "tensorflow_allclose": tf_allclose,
    "keras_allclose": keras_allclose,
    "jax_allclose": jax_allclose,
}

@given(input_data=generate_allclose_inputs())
@settings(max_examples=100, deadline=None)
def test_allclose_functions(input_data):
    run_test("test_allclose", input_data, api_functions)

if __name__ == "__main__":
    test_allclose_functions()
    finalize_results("test_allclose")
