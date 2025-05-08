from hypothesis import given, settings
from inputs.input_generator import generate_lp_pool1d_input
from functions.torch_functions import nn_torch_lp_pool1d
from functions.tf_functions import tf_lp_pool1d
from functions.keras_functions import keras_lp_pool1d
from utilities.helpers import run_test, finalize_results

api_functions = {
    "pytorch_lp_pool1d": nn_torch_lp_pool1d,
    "tensorflow_lp_pool1d": tf_lp_pool1d,
    "keras_lp_pool1d": keras_lp_pool1d,
}

@given(input_data=generate_lp_pool1d_input())
@settings(max_examples=100, deadline=None)
def test_lp_pool1d_functions(input_data):
    run_test("test_lp_pool1d", input_data, api_functions)

if __name__ == "__main__":
    test_lp_pool1d_functions()
    finalize_results("test_lp_pool1d")
