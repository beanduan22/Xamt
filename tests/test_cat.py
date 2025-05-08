from hypothesis import given, settings
from inputs.input_generator import generate_cat_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cat
from functions.tf_functions import tf_cat
from functions.chainer_functions import chainer_cat
from functions.keras_functions import keras_cat

api_functions = {
    "pytorch_cat": torch_cat,
    "tensorflow_cat": tf_cat,
    "chainer_cat": chainer_cat,
    "keras_cat": keras_cat,
}

@given(input_data=generate_cat_inputs())
@settings(max_examples=100, deadline=None)
def test_cat_functions(input_data):
    run_test("test_cat", input_data, api_functions)

if __name__ == "__main__":
    test_cat_functions()
    finalize_results("test_cat")
