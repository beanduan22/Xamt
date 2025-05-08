from hypothesis import given, settings
from inputs.input_generator import generate_test_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_abs
from functions.tf_functions import tf_abs
from functions.chainer_functions import chainer_absolute
from functions.keras_functions import keras_abs
from functions.jax_functions import jax_abs

api_functions = {
    "pytorch_abs": torch_abs,
    "tensorflow_abs": tf_abs,
    "chainer_absolute": chainer_absolute,
    "keras_abs": keras_abs,
    "jax_abs": jax_abs,
}

@given(input_data=generate_test_inputs())
@settings(max_examples=100, deadline=None)
def test_abs_functions(input_data):
    run_test("test_abs", input_data, api_functions)

if __name__ == "__main__":
    test_abs_functions()
    finalize_results("test_abs")
