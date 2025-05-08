from hypothesis import given, settings
from inputs.input_generator import generate_argmin_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_argmin
from functions.tf_functions import tf_argmin
from functions.chainer_functions import chainer_argmin
from functions.jax_functions import jax_argmin

api_functions = {
    "pytorch_argmin": torch_argmin,
    "tensorflow_argmin": tf_argmin,
    "chainer_argmin": chainer_argmin,
    "jax_argmin": jax_argmin,
}

@given(input_data=generate_argmin_inputs())
@settings(max_examples=100, deadline=None)
def test_argmin_functions(input_data):
    run_test("test_argmin", input_data, api_functions)

if __name__ == "__main__":
    test_argmin_functions()
    finalize_results("test_argmin")