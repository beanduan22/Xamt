from hypothesis import given, settings
from inputs.input_generator import generate_nn_embedding_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_nn_embedding
from functions.tf_functions import tf_nn_embedding
from functions.keras_functions import keras_nn_embedding


api_functions = {
    "pytorch_nn_embedding": torch_nn_embedding,
    "tensorflow_nn_embedding": tf_nn_embedding,
    "keras_nn_embedding": keras_nn_embedding,
}

@given(input_data=generate_nn_embedding_input())
@settings(max_examples=100, deadline=None)
def test_nn_embedding(input_data):
    run_test("test_nn_embedding", input_data, api_functions)

if __name__ == "__main__":
    test_nn_embedding()
    finalize_results("test_nn_embedding")
