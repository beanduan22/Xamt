from hypothesis import given, settings
from inputs.input_generator import generate_elu_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_elu
from functions.tf_functions import tf_elu
from functions.keras_functions import keras_elu
from functions.chainer_functions import chainer_elu
from functions.jax_functions import jax_elu

api_functions = {
    "pytorch_elu": torch_elu,
    "tensorflow_elu": tf_elu,
    "keras_elu": keras_elu,
    "chainer_elu": chainer_elu,
    "jax_elu": jax_elu,
}

@given(input_data=generate_elu_inputs())
@settings(max_examples=100, deadline=None)
def test_elu_functions(input_data):
    run_test("test_elu", input_data, api_functions)

if __name__ == "__main__":
    test_elu_functions()
    finalize_results("test_elu")
