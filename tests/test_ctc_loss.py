from hypothesis import given, settings
from inputs.input_generator import generate_ctc_loss_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_ctc_loss
from functions.tf_functions import tf_ctc_loss
from functions.chainer_functions import chainer_ctc_loss

api_functions = {
    "pytorch_ctc_loss": torch_ctc_loss,
    "tensorflow_ctc_loss": tf_ctc_loss,
    "chainer_ctc_loss": chainer_ctc_loss,
}

@given(input_data=generate_ctc_loss_inputs())
@settings(max_examples=100, deadline=None)
def test_ctc_loss_functions(input_data):
    run_test("test_ctc_loss", input_data, api_functions)

if __name__ == "__main__":
    test_ctc_loss_functions()
    finalize_results("test_ctc_loss")
