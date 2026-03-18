<h1 align="center">OemgaSqueeze</h1>

<p align="center">
  <strong>A neural network compiler stack for ultra-low-power biosensor intelligence</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/></a>
  <img src="https://img.shields.io/badge/Frontend-PyTorch-ee4c2c.svg" alt="PyTorch Frontend"/>
  <img src="https://img.shields.io/badge/Codegen-Pure%20C-00599C.svg" alt="Pure C Codegen"/>
  <img src="https://img.shields.io/badge/Target-Zephyr%20RTOS-black.svg" alt="Zephyr RTOS"/>
</p>

## Overview

**OemgaSqueeze** is a lightweight, purpose-built neural network compiler that translates PyTorch 1D CNN and Linear models into highly optimized, zero-dependency C code.

Designed for ultra-constrained edge AI, it removes the overhead of heavy third-party inference runtimes and instead statically compiles models into a standalone execution pipeline. The generated output is built to fit deeply embedded deployment settings, with a strong focus on **biosensor intelligence**, **predictable memory usage**, and **Zephyr RTOS integration**.

Rather than acting as a generic inference runtime, OemgaSqueeze follows a compile-first philosophy: take a trained model, lower it into a static embedded execution plan, and emit lean C code that is transparent, portable, and firmware-friendly.

## Why OemgaSqueeze?

Modern ML deployment tooling is often too heavy for tiny embedded systems. OemgaSqueeze is designed for the opposite regime. It is built from the ground up to prioritize:

  * **Ultra-low-power execution:** Tailored for bare-metal and RTOS environments.
  * **Predictable memory usage:** Exact bounds on RAM and Flash before you ever compile your firmware.
  * **Zero dynamic allocation:** Complete elimination of heap dependencies.
  * **Low firmware overhead:** Emits only the exact code needed for your specific model architecture.
  * **Efficient continuous inference:** Highly optimized for streaming 1D time-series data.

This makes it especially relevant for embedded systems where memory, latency, and deployment simplicity matter more than broad framework coverage.

## Key Features

  * **Zero-Dependency Bare-Metal C:** Generates pure C source and header files (e.g., `oemga_model.c`) with no external inference libraries required.
  * **No Dynamic Memory Allocation:** No `malloc`, no heap dependency, and no hidden runtime overhead.
  * **Aggressive Static Memory Planning:** Uses ping-pong activation buffers and shared scratchpads to keep SRAM usage bounded and predictable.
  * **Zephyr RTOS Ready:** Automatically generates an embedded application structure, complete with `main.c`, `CMakeLists.txt`, and `prj.conf` for straightforward Zephyr integration.
  * **Pre-Flight Hardware Cost Analysis:** Provides a breakdown of estimated Flash and RAM requirements before deployment, including model parameters, double buffers, and scratch memory.
  * **Host-to-Target Verification:** Generates a lightweight host-side verifier to compare generated C inference against the original PyTorch model using metrics such as macro F1 and logit difference reporting.

## Ideal Use Cases

OemgaSqueeze is optimized for low-power, continuous embedded inference workloads.

### Biosensor Intelligence

Designed for intelligent sensing systems where models must execute directly on-device under strict resource limits. It is highly optimized for processing continuous 1D time-series data, such as real-time EMG signal classification, directly at the edge.

### Open Embedded Platforms

Provides a seamless, highly efficient firmware generation path for custom wearable bio-interaction platforms and edge devices utilizing ultra-low-power microcontrollers like the nRF54 or nRF52 families.

### Strict RTOS Environments

A strong fit for systems where the following are critical:

  * Bounded memory usage
  * Deterministic execution
  * Minimal flash footprint
  * Transparent generated code
  * Straightforward integration into firmware stacks

## Installation

OemgaSqueeze is packaged for modern Python environments. It is recommended to install it within a virtual environment.

```bash
git clone https://github.com/yourusername/oemgasqueeze.git
cd oemgasqueeze
pip install -e .
```

## Quick Start

### Command Line Interface

You can compile a saved PyTorch model directly to a Zephyr application using the CLI:

```bash
oemga-squeeze path/to/your_model.pt --out build_zephyr/app/src/model --backend oemga_native_int8
```

### Python API

For deeper integration into your training pipeline or custom export scripts:

```python
import torch
from oemgasqueeze.core import OemgaSqueeze

# Load your trained model and example input
model = torch.load("my_1d_cnn.pt")
example_input = torch.randn(1, 1, 64)
calibration_data = torch.randn(100, 1, 64)

# Initialize the compiler
compiler = OemgaSqueeze(
    model=model,
    example_input=example_input,
    calibration_data=calibration_data,
    output_dir="zephyr_workspace/app",
    prefer_int8_inference=True
)

# Run the full compilation and codegen pipeline
artifacts = compiler.run_step5()
print(f"Generated Zephyr application in: {artifacts.output_root}")
```

## Limitations (v1.0)

To maintain its strict, lightweight footprint, the current version supports a targeted subset of operations necessary for 1D signal processing:

  * **Operations:** `Conv1d` (groups=1), `Linear`, `MaxPool1d`, `ReLU`, `Reshape` (Flatten).
  * **Topology:** Currently supports sequential models with exactly one input tensor.
