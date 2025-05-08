from hypothesis import given, settings
from inputs.input_generator import generate_leaky_relu_input
from functions.torch_functions import nn_torch_leaky_relu
from functions.tf_functions import tf_leaky_relu
from functions.keras_functions import keras_leaky_relu
from utilities.helpers import run_test, finalize_results

api_functions = {
    "pytorch_leaky_relu": nn_torch_leaky_relu,
    "tensorflow_leaky_relu": tf_leaky_relu,
    "keras_leaky_relu": keras_leaky_relu,
}

@given(input_data=generate_leaky_relu_input())
@settings(max_examples=100, deadline=None)
def test_leaky_relu_functions(input_data):
    run_test("test_leaky_relu", input_data, api_functions)

if __name__ == "__main__":
    test_leaky_relu_functions()
    finalize_results("test_leaky_relu")
