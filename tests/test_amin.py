from hypothesis import given, settings
from inputs.input_generator import generate_amin_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_amin
from functions.tf_functions import tf_amin
from functions.chainer_functions import chainer_amin
from functions.jax_functions import jax_amin

api_functions = {
    "pytorch_amin": torch_amin,
    "tensorflow_amin": tf_amin,
    "chainer_amin": chainer_amin,
    "jax_amin": jax_amin,
}

@given(input_data=generate_amin_inputs())
@settings(max_examples=100, deadline=None)
def test_amin_functions(input_data):
    run_test("test_amin", input_data, api_functions)

if __name__ == "__main__":
    test_amin_functions()
    finalize_results("test_amin")
