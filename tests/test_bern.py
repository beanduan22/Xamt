from hypothesis import given, settings
from inputs.input_generator import generate_bernoulli_inputs
from utilities.helpers import run_test, finalize_results
from functions.torch_functions import torch_bernoulli
from functions.tf_functions import tf_bernoulli
from functions.jax_functions import jax_bernoulli
from functions.chainer_functions import chainer_bernoulli
from functions.keras_functions import keras_bernoulli
import jax

api_functions = {
    "pytorch_bernoulli": torch_bernoulli,
    "tensorflow_bernoulli": tf_bernoulli,
    "jax_bernoulli": lambda x: jax_bernoulli(x, jax.random.PRNGKey(0)),
    "chainer_bernoulli": chainer_bernoulli,
    "keras_bernoulli": keras_bernoulli,
}

@given(input_data=generate_bernoulli_inputs())
@settings(max_examples=100, deadline=None)
def test_bernoulli_functions(input_data):
    run_test("test_bernoulli", input_data, api_functions)

if __name__ == "__main__":
    test_bernoulli_functions()
    finalize_results("test_bernoulli")
