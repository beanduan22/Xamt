from hypothesis import given, settings
from inputs.input_generator import generate_addmm_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_addmm
from functions.tf_functions import tf_addmm
from functions.keras_functions import keras_add
from functions.jax_functions import jax_add

api_functions = {
    "pytorch_addmm": torch_addmm,
    "tensorflow_addmm": tf_addmm,
    "keras_addmm": keras_add,
    "jax_addmm": jax_add,
}

@given(input_data=generate_addmm_inputs())
@settings(max_examples=100, deadline=None)
def test_addmm_functions(input_data):
    run_test("test_addmm", input_data, api_functions)

if __name__ == "__main__":
    test_addmm_functions()
    finalize_results("test_addmm")
