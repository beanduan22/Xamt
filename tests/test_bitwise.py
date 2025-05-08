from hypothesis import given, settings
from inputs.input_generator import generate_bitwise_and_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bitwise_and
from functions.tf_functions import tf_bitwise_and
from functions.jax_functions import jax_bitwise_and
from functions.chainer_functions import chainer_bitwise_and
from functions.keras_functions import keras_bitwise_and

api_functions = {
    "pytorch_bitwise_and": torch_bitwise_and,
    "tensorflow_bitwise_and": tf_bitwise_and,
    "jax_bitwise_and": jax_bitwise_and,
    "chainer_bitwise_and": chainer_bitwise_and,
    "keras_bitwise_and": keras_bitwise_and,
}

@given(input_data=generate_bitwise_and_inputs())
@settings(max_examples=100, deadline=None)
def test_bitwise_and_functions(input_data):
    run_test("test_bitwise_and", input_data, api_functions)

if __name__ == "__main__":
    test_bitwise_and_functions()
    finalize_results("test_bitwise_and")
