# Test Deep Learning Libraries Code

This project is designed to test and compare various deep learning libraries using a structured approach.

## Directory Structure

- `functions/`: Contains the functions for different deep learning libraries.
- `inputs/`: Contains the input generation strategy.
- `run_tasks/`: Contains the task scheduler for CPU and GPU tasks, and the main test setup script.
- `tests/`: Contains the individual test scripts for various APIs.
- `utilities/`: Contains common imports, counters, and summary utilities.

## How to Run

1. Generate test inputs using the scripts in `inputs/`.
2. Setup and run tests using `run_tasks/test_setup.py`.
3. Log results and execution times using the scripts in `outputs/`.
4. Summarize the results using the utilities in `utilities/`.

## Requirements

- Python 3.9.20
- PyTorch 2.2.1
- TensorFlow 2.16.1
- Chainer 7.8.1
- Keras 3.6.0
- JAX 0.4.26
- Numpy 1.26.4
- SciPy 1.13.1
