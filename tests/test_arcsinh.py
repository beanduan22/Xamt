from hypothesis import given, settings
from inputs.input_generator import generate_arcsinh_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_arcsinh
from functions.tf_functions import tf_arcsinh
from functions.chainer_functions import chainer_arcsinh
from functions.jax_functions import jax_arcsinh

api_functions = {
    "pytorch_arcsinh": torch_arcsinh,
    "tensorflow_arcsinh": tf_arcsinh,
    "chainer_arcsinh": chainer_arcsinh,
    "jax_arcsinh": jax_arcsinh,
}

@given(input_data=generate_arcsinh_inputs())
@settings(max_examples=100, deadline=None)
def test_arcsinh_functions(input_data):
    run_test("test_arcsinh", input_data, api_functions)

if __name__ == "__main__":
    test_arcsinh_functions()
    finalize_results("test_arcsinh")
