from hypothesis import given, settings
from inputs.input_generator import generate_cumsum_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_cumsum
from functions.tf_functions import tf_cumsum
from functions.chainer_functions import chainer_cumsum
from functions.keras_functions import keras_cumsum

api_functions = {
    "pytorch_cumsum": torch_cumsum,
    "tensorflow_cumsum": tf_cumsum,
    "chainer_cumsum": chainer_cumsum,
    "keras_cumsum": keras_cumsum,
}

@given(input_data=generate_cumsum_inputs())
@settings(max_examples=100, deadline=None)
def test_cumsum_functions(input_data):
    run_test("test_cumsum", input_data, api_functions)

if __name__ == "__main__":
    test_cumsum_functions()
    finalize_results("test_cumsum")
