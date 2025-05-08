from hypothesis import given, settings
from inputs.input_generator import generate_log_softmax_input
from functions.torch_functions import nn_torch_log_softmax
from functions.tf_functions import tf_log_softmax
from functions.keras_functions import keras_log_softmax
from functions.chainer_functions import chainer_log_softmax
from utilities.helpers import run_test, finalize_results

api_functions = {
    "pytorch_log_softmax": nn_torch_log_softmax,
    "tensorflow_log_softmax": tf_log_softmax,
    "keras_log_softmax": keras_log_softmax,
    "chainer_log_softmax": chainer_log_softmax,
}

@given(input_data=generate_log_softmax_input())
@settings(max_examples=100, deadline=None)
def test_log_softmax_functions(input_data):
    run_test("test_log_softmax", input_data, api_functions)

if __name__ == "__main__":
    test_log_softmax_functions()
    finalize_results("test_log_softmax")
