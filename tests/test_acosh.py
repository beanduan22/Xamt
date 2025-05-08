from hypothesis import given, settings
from inputs.input_generator import acosh_input_strategy
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_acosh
from functions.tf_functions import tf_acosh
from functions.chainer_functions import chainer_acosh
from functions.keras_functions import keras_acosh
from functions.jax_functions import jax_acosh

api_functions = {
    "pytorch_acosh": torch_acosh,
    "tensorflow_acosh": tf_acosh,
    "chainer_acosh": chainer_acosh,
    "keras_acosh": keras_acosh,
    "jax_acosh": jax_acosh,
}

@given(input_data=acosh_input_strategy())
@settings(max_examples=1000, deadline=None)
def test_acosh_functions(input_data):
    run_test("test_acosh", input_data, api_functions)

if __name__ == "__main__":
    test_acosh_functions()
    finalize_results("test_acosh")
