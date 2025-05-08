import torch
import tensorflow as tf
import chainer

from hypothesis import given, settings
import hypothesis.strategies as st

import jax

# PyTorch
def torch_are_deterministic_algorithms_enabled():
    return torch.are_deterministic_algorithms_enabled()

# TensorFlow
def tf_executing_eagerly():
    return tf.executing_eagerly()

# Chainer
def chainer_is_debug():
    return chainer.is_debug()

# JAX
def jax_config_omnistaging_enabled():
    return jax.config.omnistaging

# 使用Hypothesis进行fuzzing测试
@given(dummy_input=st.integers())
@settings(deadline=None)
def test_framework_functions(dummy_input):
    # 获取各个框架的输出结果
    torch_output = torch_are_deterministic_algorithms_enabled()
    tf_output = tf_executing_eagerly()
    chainer_output = chainer_is_debug()
    jax_output = jax_config_omnistaging_enabled()

    # 检查结果是否一致，如果不一致则打印
    if not (torch_output == tf_output == chainer_output == jax_output):
        print("Inconsistent result found!")
        print("PyTorch Output:", torch_output)
        print("TensorFlow Output:", tf_output)
        print("Chainer Output:", chainer_output)
        print("JAX Output:", jax_output)

# 运行测试
if __name__ == "__main__":
    test_framework_functions()
