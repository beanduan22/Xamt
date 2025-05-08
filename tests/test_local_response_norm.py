from hypothesis import given, settings
from inputs.input_generator import generate_local_response_norm_input
from functions.torch_functions import nn_torch_local_response_norm
from functions.tf_functions import tf_local_response_norm
from functions.keras_functions import keras_local_response_norm
from functions.chainer_functions import chainer_local_response_norm
from utilities.helpers import run_test, finalize_results

api_functions = {
    "pytorch_local_response_norm": nn_torch_local_response_norm,
    "tensorflow_local_response_norm": tf_local_response_norm,
    "keras_local_response_norm": keras_local_response_norm,
    "chainer_local_response_norm": chainer_local_response_norm,
}

@given(input_data=generate_local_response_norm_input())
@settings(max_examples=100, deadline=None)
def test_local_response_norm_functions(input_data):
    run_test("test_local_response_norm", input_data, api_functions)

if __name__ == "__main__":
    test_local_response_norm_functions()
    finalize_results("test_local_response_norm")
