from hypothesis import given, settings
from inputs.input_generator import generate_cross_entropy_loss_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cross_entropy_loss
from functions.tf_functions import tf_cross_entropy_loss
from functions.keras_functions import keras_cross_entropy_loss
from functions.chainer_functions import chainer_cross_entropy_loss
from functions.jax_functions import jax_cross_entropy_loss

api_functions = {
    "pytorch_cross_entropy_loss": torch_cross_entropy_loss,
    "tensorflow_cross_entropy_loss": tf_cross_entropy_loss,
    "keras_cross_entropy_loss": keras_cross_entropy_loss,
    "chainer_cross_entropy_loss": chainer_cross_entropy_loss,
    "jax_cross_entropy_loss": jax_cross_entropy_loss,
}

@given(input_data=generate_cross_entropy_loss_inputs())
@settings(max_examples=100, deadline=None)
def test_cross_entropy_loss_functions(input_data):
    run_test("test_cross_entropy_loss", input_data, api_functions)

if __name__ == "__main__":
    test_cross_entropy_loss_functions()
    finalize_results("test_cross_entropy_loss")
