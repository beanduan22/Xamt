from hypothesis import given, settings
from inputs.input_generator import generate_digamma_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_digamma
from functions.tf_functions import tf_digamma
from functions.chainer_functions import chainer_digamma
from functions.jax_functions import jax_digamma

api_functions = {
    "pytorch_digamma": torch_digamma,
    "tensorflow_digamma": tf_digamma,
    "chainer_digamma": chainer_digamma,
    "jax_digamma": jax_digamma,
}

@given(input_data=generate_digamma_inputs())
@settings(max_examples=100, deadline=None)
def test_digamma_functions(input_data):
    run_test("test_digamma", input_data, api_functions)

if __name__ == "__main__":
    test_digamma_functions()
    finalize_results("test_digamma")
