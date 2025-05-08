from hypothesis import given, settings
from inputs.input_generator import generate_concat_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_concat
from functions.tf_functions import tf_concat
from functions.keras_functions import keras_concat
from functions.chainer_functions import chainer_concat
from functions.jax_functions import jax_concat

api_functions = {
    "pytorch_concat": torch_concat,
    "tensorflow_concat": tf_concat,
    "keras_concat": keras_concat,
    "chainer_concat": chainer_concat,
    "jax_concat": jax_concat,
}

@given(input_data=generate_concat_inputs())
@settings(max_examples=100, deadline=None)
def test_concat_functions(input_data):
    run_test("test_concat", input_data, api_functions)

if __name__ == "__main__":
    test_concat_functions()
    finalize_results("test_concat")
