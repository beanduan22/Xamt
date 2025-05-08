from hypothesis import given, settings
from inputs.input_generator import generate_ceil_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_deg2rad
from functions.tf_functions import tf_deg2rad
from functions.chainer_functions import chainer_deg2rad
from functions.jax_functions import jax_deg2rad

api_functions = {
    "pytorch_deg2rad": torch_deg2rad,
    "tensorflow_deg2rad": tf_deg2rad,
    "chainer_deg2rad": chainer_deg2rad,
    "jax_deg2rad": jax_deg2rad,
}

@given(input_data=generate_ceil_inputs())
@settings(max_examples=100, deadline=None)
def test_deg2rad_functions(input_data):
    run_test("test_deg2rad", input_data, api_functions)

if __name__ == "__main__":
    test_deg2rad_functions()
    finalize_results("test_deg2rad")
