from hypothesis import given, settings
from inputs.input_generator import generate_nn_data_parallel_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_data_parallel
from functions.tf_functions import nn_tf_data_parallel
from functions.chainer_functions import nn_chainer_data_parallel
from functions.keras_functions import nn_keras_data_parallel
from functions.jax_functions import nn_jax_data_parallel

api_functions = {
    "pytorch_data_parallel": nn_torch_data_parallel,
    "tensorflow_data_parallel": nn_tf_data_parallel,
    "chainer_data_parallel": nn_chainer_data_parallel,
    "keras_data_parallel": nn_keras_data_parallel,
    "jax_data_parallel": nn_jax_data_parallel,
}

@given(input_data=generate_nn_data_parallel_input())
@settings(max_examples=100, deadline=None)
def test_data_parallel_functions(input_data):
    run_test("test_data_parallel", input_data, api_functions)

if __name__ == "__main__":
    test_data_parallel_functions()
    finalize_results("test_data_parallel")
