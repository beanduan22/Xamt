from hypothesis import given, settings
from inputs.input_generator import generate_lp_pool2d_input
from functions.torch_functions import nn_torch_lp_pool2d
from functions.tf_functions import tf_lp_pool2d
from functions.keras_functions import keras_lp_pool2d
from utilities.helpers import run_test, finalize_results

api_functions = {
    "pytorch_lp_pool2d": nn_torch_lp_pool2d,
    "tensorflow_lp_pool2d": tf_lp_pool2d,
    "keras_lp_pool2d": keras_lp_pool2d,
}

@given(input_data=generate_lp_pool2d_input())
@settings(max_examples=100, deadline=None)
def test_lp_pool2d_functions(input_data):
    run_test("test_lp_pool2d", input_data, api_functions)

if __name__ == "__main__":
    test_lp_pool2d_functions()
    finalize_results("test_lp_pool2d")
