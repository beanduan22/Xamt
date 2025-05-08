from hypothesis import given, settings
from inputs.input_generator import generate_nn_ctc_loss_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_ctc_loss
from functions.tf_functions import nn_tf_ctc_loss
from functions.chainer_functions import nn_chainer_ctc_loss
from functions.keras_functions import nn_keras_ctc_loss
from functions.jax_functions import nn_jax_ctc_loss

api_functions = {
    "pytorch_ctc_loss": nn_torch_ctc_loss,
    "tensorflow_ctc_loss": nn_tf_ctc_loss,
    "chainer_ctc_loss": nn_chainer_ctc_loss,
    "keras_ctc_loss": nn_keras_ctc_loss,
    "jax_ctc_loss": nn_jax_ctc_loss,
}

@given(input_data=generate_nn_ctc_loss_input())
@settings(max_examples=100, deadline=None)
def test_ctc_loss_functions(input_data):
    run_test("test_ctc_loss", input_data, api_functions)

if __name__ == "__main__":
    test_ctc_loss_functions()
    finalize_results("test_ctc_loss")
