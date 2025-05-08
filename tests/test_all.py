from hypothesis import given, settings
from inputs.input_generator import generate_all_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_all
from functions.tf_functions import tf_all
from functions.keras_functions import keras_all
from functions.chainer_functions import chainer_all
from functions.jax_functions import jax_all

api_functions = {
    "pytorch_all": torch_all,
    "tensorflow_all": tf_all,
    "keras_all": keras_all,
    "chainer_all": chainer_all,
    "jax_all": jax_all,
}

@given(input_data=generate_all_inputs())
@settings(max_examples=100, deadline=None)
def test_all_functions(input_data):
    run_test("test_all", (input_data,), api_functions)

if __name__ == "__main__":
    test_all_functions()
    finalize_results("test_all")
