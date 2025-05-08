from hypothesis import given, settings
from inputs.input_generator import generate_chunk_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_chunk
from functions.tf_functions import tf_chunk
from functions.chainer_functions import chainer_chunk
from functions.keras_functions import keras_chunk

api_functions = {
    "pytorch_chunk": torch_chunk,
    "tensorflow_chunk": tf_chunk,
    "chainer_chunk": chainer_chunk,
    "keras_chunk": keras_chunk,
}

@given(input_data=generate_chunk_inputs())
@settings(max_examples=100, deadline=None)
def test_chunk_functions(input_data):
    run_test("test_chunk", input_data, api_functions)

if __name__ == "__main__":
    test_chunk_functions()
    finalize_results("test_chunk")
