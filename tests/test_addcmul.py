from hypothesis import given, settings
from inputs.input_generator import addcmul_input_strategy
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_addcmul
from functions.tf_functions import tf_addcmul
from functions.chainer_functions import chainer_addcmul
from functions.jax_functions import jax_addcmul

api_functions = {
    "pytorch_addcmul": torch_addcmul,
    "tensorflow_addcmul": tf_addcmul,
    "chainer_addcmul": chainer_addcmul,
    "jax_addcmul": jax_addcmul,
}

@given(input_data=addcmul_input_strategy())
@settings(max_examples=1000, deadline=None)
def test_addcmul_functions(input_data):
    run_test("test_addcmul", input_data, api_functions)

if __name__ == "__main__":
    test_addcmul_functions()
    finalize_results("test_addcmul")
