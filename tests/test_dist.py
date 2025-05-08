from hypothesis import given, settings
from inputs.input_generator import generate_dist_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_dist
from functions.tf_functions import tf_dist
from functions.chainer_functions import chainer_dist
from functions.jax_functions import jax_dist

api_functions = {
    "pytorch_dist": torch_dist,
    "tensorflow_dist": tf_dist,
    "chainer_dist": chainer_dist,
    "jax_dist": jax_dist,
}

@given(input_data=generate_dist_inputs())
@settings(max_examples=100, deadline=None)
def test_dist_functions(input_data):
    run_test("test_dist", input_data, api_functions)

if __name__ == "__main__":
    test_dist_functions()
    finalize_results("test_dist")
