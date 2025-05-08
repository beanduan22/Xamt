from hypothesis import given, settings
from inputs.input_generator import generate_cosine_embedding_loss_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cosine_embedding_loss
from functions.tf_functions import tf_cosine_embedding_loss
from functions.keras_functions import keras_cosine_embedding_loss
from functions.chainer_functions import chainer_cosine_embedding_loss
from functions.jax_functions import jax_cosine_embedding_loss

api_functions = {
    "pytorch_cosine_embedding_loss": torch_cosine_embedding_loss,
    "tensorflow_cosine_embedding_loss": tf_cosine_embedding_loss,
    "keras_cosine_embedding_loss": keras_cosine_embedding_loss,
    "chainer_cosine_embedding_loss": chainer_cosine_embedding_loss,
    "jax_cosine_embedding_loss": jax_cosine_embedding_loss,
}

@given(input_data=generate_cosine_embedding_loss_inputs())
@settings(max_examples=100, deadline=None)
def test_cosine_embedding_loss_functions(input_data):
    run_test("test_cosine_embedding_loss", input_data, api_functions)

if __name__ == "__main__":
    test_cosine_embedding_loss_functions()
    finalize_results("test_cosine_embedding_loss")
