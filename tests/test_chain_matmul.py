from hypothesis import given, settings
from inputs.input_generator import generate_chain_matmul_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_chain_matmul
from functions.tf_functions import tf_chain_matmul
from functions.chainer_functions import chainer_chain_matmul

api_functions = {
    "pytorch_chain_matmul": torch_chain_matmul,
    "tensorflow_chain_matmul": tf_chain_matmul,
    "chainer_chain_matmul": chainer_chain_matmul,
}

@given(input_data=generate_chain_matmul_inputs())
@settings(max_examples=100, deadline=None)
def test_chain_matmul_functions(input_data):
    run_test("test_chain_matmul", input_data, api_functions)

if __name__ == "__main__":
    test_chain_matmul_functions()
    finalize_results("test_chain_matmul")
