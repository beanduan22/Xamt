from hypothesis import given, settings
from inputs.input_generator import generate_clone_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_clone
from functions.tf_functions import tf_clone
from functions.chainer_functions import chainer_clone
from functions.keras_functions import keras_clone

api_functions = {
    "pytorch_clone": torch_clone,
    "tensorflow_clone": tf_clone,
    "chainer_clone": chainer_clone,
    "keras_clone": keras_clone,
}

@given(input_data=generate_clone_inputs())
@settings(max_examples=100, deadline=None)
def test_clone_functions(input_data):
    run_test("test_clone", input_data, api_functions)

if __name__ == "__main__":
    test_clone_functions()
    finalize_results("test_clone")
