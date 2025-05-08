from hypothesis import given, settings
from inputs.input_generator import generate_BFloat16Storage_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_BFloat16Storage
from functions.chainer_functions import chainer_empty_bfloat16

api_functions = {
    "pytorch_BFloat16Storage": torch_BFloat16Storage,
    "chainer_empty_bfloat16": chainer_empty_bfloat16,
}

@given(size=generate_BFloat16Storage_inputs())
@settings(max_examples=100, deadline=None)
def test_BFloat16Storage_functions(size):
    run_test("test_BFloat16Storage", (size,), api_functions)

if __name__ == "__main__":
    test_BFloat16Storage_functions()
    finalize_results("test_BFloat16Storage")
