from hypothesis import given, settings
from inputs.input_generator import generate_dropout_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_dropout
from functions.tf_functions import tf_dropout
from functions.keras_functions import keras_dropout
from functions.chainer_functions import chainer_dropout
from functions.jax_functions import jax_dropout

api_functions = {
    "pytorch_dropout": torch_dropout,
    "tensorflow_dropout": tf_dropout,
    "keras_dropout": keras_dropout,
    "chainer_dropout": chainer_dropout,
    "jax_dropout": jax_dropout,
}

@given(input_data=generate_dropout_inputs())
@settings(max_examples=100, deadline=None)
def test_dropout_functions(input_data):
    run_test("test_dropout", input_data, api_functions)

if __name__ == "__main__":
    test_dropout_functions()
    finalize_results("test_dropout")
