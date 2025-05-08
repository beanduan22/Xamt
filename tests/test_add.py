from hypothesis import given, settings
from inputs.input_generator import add_input_strategy
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_add
from functions.tf_functions import tf_add
from functions.chainer_functions import chainer_add
from functions.keras_functions import keras_add
from functions.jax_functions import jax_add

api_functions = {
    "pytorch_add": torch_add,
    "tensorflow_add": tf_add,
    "chainer_add": chainer_add,
    "keras_add": keras_add,
    "jax_add": jax_add,
}

@given(input_data=add_input_strategy())
@settings(max_examples=2000, deadline=None)
def test_add_functions(input_data):
    run_test("test_add", input_data, api_functions)

if __name__ == "__main__":
    test_add_functions()
    finalize_results("test_add")
