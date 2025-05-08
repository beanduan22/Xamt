from hypothesis import given, settings
from inputs.input_generator import generate_layer_norm_input
from functions.torch_functions import nn_torch_layer_norm
from functions.tf_functions import tf_layer_norm
from functions.keras_functions import keras_layer_norm
from utilities.helpers import run_test, finalize_results

api_functions = {
    "pytorch_layer_norm": nn_torch_layer_norm,
    "tensorflow_layer_norm": tf_layer_norm,
    "keras_layer_norm": keras_layer_norm,
}

@given(input_data=generate_layer_norm_input())
@settings(max_examples=100, deadline=None)
def test_layer_norm_functions(input_data):
    run_test("test_layer_norm", input_data, api_functions)

if __name__ == "__main__":
    test_layer_norm_functions()
    finalize_results("test_layer_norm")
