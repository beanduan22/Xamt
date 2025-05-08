from hypothesis import given, settings
from inputs.input_generator import generate_celu_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_celu
from functions.tf_functions import tf_celu
from functions.chainer_functions import chainer_celu
from functions.jax_functions import jax_celu

api_functions = {
    "pytorch_celu": torch_celu,
    "tensorflow_celu": tf_celu,
    "chainer_celu": chainer_celu,
    "jax_celu": jax_celu,
}

@given(input_data=generate_celu_inputs())
@settings(max_examples=100, deadline=None)
def test_celu_functions(input_data):
    run_test("test_celu", input_data, api_functions)

if __name__ == "__main__":
    test_celu_functions()
    finalize_results("test_celu")
