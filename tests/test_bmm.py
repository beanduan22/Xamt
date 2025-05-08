from hypothesis import given, settings
from inputs.input_generator import generate_bmm_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bmm
from functions.tf_functions import tf_bmm
from functions.chainer_functions import chainer_bmm

api_functions = {
    "pytorch_bmm": torch_bmm,
    "tensorflow_bmm": tf_bmm,
    "chainer_bmm": chainer_bmm,
}

@given(input_data=generate_bmm_inputs())
@settings(max_examples=100, deadline=None)
def test_bmm_functions(input_data):
    run_test("test_bmm", input_data, api_functions)

if __name__ == "__main__":
    test_bmm_functions()
    finalize_results("test_bmm")
