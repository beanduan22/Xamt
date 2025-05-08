from hypothesis import given, settings
from inputs.input_generator import generate_cosine_similarity_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cosine_similarity
from functions.tf_functions import tf_cosine_similarity
from functions.keras_functions import keras_cosine_similarity
from functions.chainer_functions import chainer_cosine_similarity
from functions.jax_functions import jax_cosine_similarity

api_functions = {
    "pytorch_cosine_similarity": torch_cosine_similarity,
    "tensorflow_cosine_similarity": tf_cosine_similarity,
    "keras_cosine_similarity": keras_cosine_similarity,
    "chainer_cosine_similarity": chainer_cosine_similarity,
    "jax_cosine_similarity": jax_cosine_similarity,
}

@given(input_data=generate_cosine_similarity_inputs())
@settings(max_examples=100, deadline=None)
def test_cosine_similarity_functions(input_data):
    run_test("test_cosine_similarity", input_data, api_functions)

if __name__ == "__main__":
    test_cosine_similarity_functions()
    finalize_results("test_cosine_similarity")
