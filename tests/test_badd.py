from hypothesis import given, settings
from inputs.input_generator import generate_baddbmm_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_baddbmm
from functions.tf_functions import tf_baddbmm
from functions.jax_functions import jax_baddbmm
from functions.chainer_functions import chainer_baddbmm
from functions.keras_functions import keras_baddbmm

api_functions = {
    "pytorch_baddbmm": torch_baddbmm,
    "tensorflow_baddbmm": tf_baddbmm,
    "jax_baddbmm": jax_baddbmm,
    "chainer_baddbmm": chainer_baddbmm,
    "keras_baddbmm": keras_baddbmm,
}

@given(input_data=generate_baddbmm_inputs())
@settings(max_examples=100, deadline=None)
def test_baddbmm_functions(input_data):
    run_test("test_baddbmm", input_data, api_functions)

if __name__ == "__main__":
    test_baddbmm_functions()
    finalize_results("test_baddbmm")
