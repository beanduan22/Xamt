from hypothesis import given, settings
from inputs.input_generator import generate_bce_loss_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bce_loss
from functions.tf_functions import tf_bce_loss
from functions.chainer_functions import chainer_bce_loss

api_functions = {
    "pytorch_bce_loss": torch_bce_loss,
    "tensorflow_bce_loss": tf_bce_loss,
    "chainer_bce_loss": chainer_bce_loss,
}

@given(input_data=generate_bce_loss_inputs())
@settings(max_examples=100, deadline=None)
def test_bce_loss_functions(input_data):
    run_test("test_bce_loss", input_data, api_functions)

if __name__ == "__main__":
    test_bce_loss_functions()
    finalize_results("test_bce_loss")
