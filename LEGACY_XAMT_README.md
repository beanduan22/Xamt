# Xamt: Cross-Framework API Matching for Testing Deep Learning Libraries

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16910387.svg)](https://doi.org/10.5281/zenodo.16910387)
[![GitHub release](https://img.shields.io/github/v/release/beanduan22/Xamt)](https://github.com/beanduan22/Xamt/releases)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

**Xamt** is a cross-framework testing toolkit for deep-learning libraries.
It aligns *functionally equivalent* APIs across **PyTorch**, **TensorFlow/Keras**, **JAX**, and **Chainer**, generates valid inputs, and runs differential tests to uncover behavioral inconsistencies and potential defects. The artifact includes:

* A normalized **API function layer** for each framework
* An **input generator** with shape/type/range constraints
* A lightweight **task runner** for CPU/GPU batches
* A **test suite** of per-API checks
* **Utilities** for logging, counters, and result summaries

---

## Directory Structure

```
Xamt/
├─ functions/                # Framework-specific callable shims
│  ├─ torch_functions.py
│  ├─ tf_functions.py
│  ├─ keras_functions.py
│  ├─ jax_functions.py
│  └─ chainer_functions.py
│
├─ inputs/                   # Input generation strategy
│  └─ input_generator.py
│
├─ run_tasks/                # Orchestration for experiments
│  ├─ test_setup.py          # Main entry for configuring runs
│  └─ task_scheduler.py      # Simple CPU/GPU scheduler
│
├─ tests/
│  ├─ test_add.py
│  ├─ test_abs.py
│  ├─ test_adaptive_avg_pool2d.py
│  └─ ...
│
├─ utilities/
│  ├─ common_imports.py
│  ├─ counters.py
│  ├─ logger.py
│  └─ summary.py
│
├─ README.md
└─ try.py                    # Scratch/demo scripts (optional)
```

---

## Installation

Xamt targets Python **3.9.20** with the following core deps:

* PyTorch **2.2.1**
* TensorFlow **2.16.1**
* Keras **3.6.0**
* JAX **0.4.26**
* Chainer **7.8.1**
* NumPy **1.26.4**
* SciPy **1.13.1**

You can use `pip` (recommended in a fresh virtualenv):

```bash
python3.9 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Example minimal installs (adjust per your platform/CUDA):
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
pip install "tensorflow==2.16.1"
pip install "keras==3.6.0"
pip install "jax[cuda12_local]==0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "chainer==7.8.1" "numpy==1.26.4" "scipy==1.13.1"
```

> ⚠️ JAX/TensorFlow/PyTorch GPU wheels depend on your CUDA/cuDNN stack. If you’re CPU-only, install the CPU wheels for each framework.

---

## Quick Start

1. **Generate inputs**

```bash
python -m inputs.input_generator \
  --out ./generated_inputs \
  --num-cases 100
```

2. **Configure and run tests**

```bash
# Edit run_tasks/test_setup.py to choose:
#   - frameworks to compare
#   - API list / test selection
#   - device (cpu / cuda)
python -m run_tasks.test_setup --device cpu --inputs ./generated_inputs
```

3. **View logs & summary**

* Logs are written via `utilities/logger.py` (default: `./logs/`).
* Run a summary after tests:

```bash
python -c "from utilities.summary import summarize; summarize('./logs', out_csv='./summary.csv')"
```

---

## How to Cite

If you use **Xamt** in academic work, please cite the paper:

**BibTeX**

```bibtex
@INPROCEEDINGS{Xamt,
  author={Duan, Bin and Dong, Ruican and Dong, Naipeng and Kim, Dan Dongseong and Yang, Guowei},
  booktitle={2025 IEEE 36th International Symposium on Software Reliability Engineering (ISSRE)},
  title={XAMT: Cross-Framework API Matching for Testing Deep Learning Libraries},
  year={2025},
  pages={191-202},
  keywords={Fuzzing;Software reliability;Testing;Deep Learning Libraries},
  doi={10.1109/ISSRE66568.2025.00030}}
```


---
