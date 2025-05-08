from hypothesis import given, settings
from inputs.input_generator import addcdiv_input_strategy
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_addcdiv
from functions.tf_functions import tf_addcdiv
from functions.chainer_functions import chainer_addcdiv
from functions.jax_functions import jax_addcdiv

api_functions = {
    "pytorch_addcdiv": torch_addcdiv,
    "tensorflow_addcdiv": tf_addcdiv,
    "chainer_addcdiv": chainer_addcdiv,
    "jax_addcdiv": jax_addcdiv,
}

@given(input_data=addcdiv_input_strategy())
@settings(max_examples=1000, deadline=None)
def test_addcdiv_functions(input_data):
    run_test("test_addcdiv", input_data, api_functions)

if __name__ == "__main__":
    test_addcdiv_functions()
    finalize_results("test_addcdiv")
