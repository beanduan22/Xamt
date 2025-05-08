from hypothesis import given, settings
from inputs.input_generator import generate_diag_embed_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_diag_embed
from functions.tf_functions import tf_diag_embed
from functions.chainer_functions import chainer_diag_embed

api_functions = {
    "pytorch_diag_embed": torch_diag_embed,
    "tensorflow_diag_embed": tf_diag_embed,
    "chainer_diag_embed": chainer_diag_embed,
}

@given(input_data=generate_diag_embed_inputs())
@settings(max_examples=100, deadline=None)
def test_diag_embed_functions(input_data):
    run_test("test_diag_embed", input_data, api_functions)

if __name__ == "__main__":
    test_diag_embed_functions()
    finalize_results("test_diag_embed")
