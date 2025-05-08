from hypothesis import given, settings
from inputs.input_generator import generate_conj_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_conj
from functions.tf_functions import tf_conj
from functions.chainer_functions import chainer_conj
from functions.jax_functions import jax_conj

api_functions = {
    "pytorch_conj": torch_conj,
    "tensorflow_conj": tf_conj,
    "chainer_conj": chainer_conj,
    "jax_conj": jax_conj,
}

@given(input_data=generate_conj_inputs())
@settings(max_examples=100, deadline=None)
def test_conj_functions(input_data):
    run_test("test_conj", input_data, api_functions)

if __name__ == "__main__":
    test_conj_functions()
    finalize_results("test_conj")
