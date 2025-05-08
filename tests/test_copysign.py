from hypothesis import given, settings
from inputs.input_generator import generate_copysign_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_copysign
from functions.tf_functions import tf_copysign
from functions.chainer_functions import chainer_copysign

api_functions = {
    "pytorch_copysign": torch_copysign,
    "tensorflow_copysign": tf_copysign,
    "chainer_copysign": chainer_copysign,
}

@given(input_data=generate_copysign_inputs())
@settings(max_examples=100, deadline=None)
def test_copysign_functions(input_data):
    run_test("test_copysign", input_data, api_functions)

if __name__ == "__main__":
    test_copysign_functions()
    finalize_results("test_copysign")
