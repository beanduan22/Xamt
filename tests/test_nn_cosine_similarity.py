from hypothesis import given, settings
from inputs.input_generator import generate_nn_cosine_similarity_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_cosine_similarity
from functions.tf_functions import nn_tf_cosine_similarity
from functions.chainer_functions import nn_chainer_cosine_similarity
from functions.keras_functions import nn_keras_cosine_similarity
from functions.jax_functions import nn_jax_cosine_similarity

api_functions = {
    "pytorch_cosine_similarity": nn_torch_cosine_similarity,
    "tensorflow_cosine_similarity": nn_tf_cosine_similarity,
    "chainer_cosine_similarity": nn_chainer_cosine_similarity,
    "keras_cosine_similarity": nn_keras_cosine_similarity,
    "jax_cosine_similarity": nn_jax_cosine_similarity,
}

@given(input_data=generate_nn_cosine_similarity_input())
@settings(max_examples=100, deadline=None)
def test_cosine_similarity_functions(input_data):
    run_test("test_cosine_similarity", input_data, api_functions)

if __name__ == "__main__":
    test_cosine_similarity_functions()
    finalize_results("test_cosine_similarity")
