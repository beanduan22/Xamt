from hypothesis import given, settings
from inputs.input_generator import generate_nn_feature_alpha_dropout_input
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_nn_feature_alpha_dropout
from functions.tf_functions import tf_nn_feature_alpha_dropout
from functions.keras_functions import keras_nn_feature_alpha_dropout
from functions.chainer_functions import chainer_nn_feature_alpha_dropout
from functions.jax_functions import jax_nn_feature_alpha_dropout

api_functions = {
    "pytorch_nn_feature_alpha_dropout": torch_nn_feature_alpha_dropout,
    "tensorflow_nn_feature_alpha_dropout": tf_nn_feature_alpha_dropout,
    "keras_nn_feature_alpha_dropout": keras_nn_feature_alpha_dropout,
    "chainer_nn_feature_alpha_dropout": chainer_nn_feature_alpha_dropout,
    "jax_nn_feature_alpha_dropout": jax_nn_feature_alpha_dropout,
}

@given(input_data=generate_nn_feature_alpha_dropout_input())
@settings(max_examples=100, deadline=None)
def test_nn_feature_alpha_dropout(input_data):
    run_test("test_nn_feature_alpha_dropout", input_data, api_functions)

if __name__ == "__main__":
    test_nn_feature_alpha_dropout()
    finalize_results("test_nn_feature_alpha_dropout")
