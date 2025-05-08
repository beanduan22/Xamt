from hypothesis import given, settings
from inputs.input_generator import generate_embedding_bag_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_embedding_bag
from functions.tf_functions import tf_embedding_bag
from functions.chainer_functions import chainer_embedding_bag

api_functions = {
    "pytorch_embedding_bag": torch_embedding_bag,
    "tensorflow_embedding_bag": tf_embedding_bag,
    "chainer_embedding_bag": chainer_embedding_bag,
}

@given(input_data=generate_embedding_bag_inputs())
@settings(max_examples=100, deadline=None)
def test_embedding_bag_functions(input_data):
    run_test("test_embedding_bag", input_data, api_functions)

if __name__ == "__main__":
    test_embedding_bag_functions()
    finalize_results("test_embedding_bag")
