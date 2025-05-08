from hypothesis import given, settings
from inputs.input_generator import generate_nn_dropout3d_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import nn_torch_dropout3d
from functions.tf_functions import nn_tf_dropout3d
from functions.chainer_functions import nn_chainer_dropout3d
from functions.keras_functions import nn_keras_dropout3d
from functions.jax_functions import nn_jax_dropout3d

api_functions = {
    "pytorch_dropout3d": nn_torch_dropout3d,
    "tensorflow_dropout3d": nn_tf_dropout3d,
    "chainer_dropout3d": nn_chainer_dropout3d,
    "keras_dropout3d": nn_keras_dropout3d,
    "jax_dropout3d": nn_jax_dropout3d,
}

@given(input_data=generate_nn_dropout3d_input())
@settings(max_examples=100, deadline=None)
def test_dropout3d_functions(input_data):
    run_test("test_dropout3d", input_data, api_functions)

if __name__ == "__main__":
    test_dropout3d_functions()
    finalize_results("test_dropout3d")
