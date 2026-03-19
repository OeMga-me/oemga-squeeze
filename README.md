<h1 align="center">OeMga.me Squeeze</h1>

<p align="center">
  <strong>A neural network compiler stack for ultra-low-power biosensor intelligence</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/></a>
  <img src="https://img.shields.io/badge/Frontend-PyTorch-ee4c2c.svg" alt="PyTorch Frontend"/>
  <img src="https://img.shields.io/badge/Codegen-Pure%20C-00599C.svg" alt="Pure C Codegen"/>
  <img src="https://img.shields.io/badge/Target-Zephyr%20RTOS-black.svg" alt="Zephyr RTOS"/>
</p>

<div align="center">
  <table>
    <tr>
      <td>
<img width="600" height="1352" alt="Untitled design (1)" src="https://github.com/user-attachments/assets/929fb8c4-9cbe-4fae-b570-53d4baa5a173" />
      </td>
    </tr>
  </table>
</div>

## Overview

**OemgaSqueeze** compiles PyTorch models into lean, firmware-native C for ultra-constrained edge devices.

No heavyweight runtime. No dependency bloat. Just deterministic neural inference, lowered into a static execution path built for real embedded hardware.

Designed for biosignal and sensor intelligence, it brings a compile-first approach to edge AI: transparent, portable, zero-dependency code ready for Zephyr-class deployment.

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

## Installation

OemgaSqueeze is packaged for modern Python environments. It is highly recommended to install it within a virtual environment.

```bash
git clone https://github.com/OeMga-me/oemga-squeeze.git
cd oemga-squeeze
pip install -e .
```

## End-to-End Example

Because accurate INT8 quantization requires real calibration data, OemgaSqueeze relies on a clean Python API rather than a generic command-line interface.

Below is a complete, self-contained script. You can copy this into a file named `example.py` and run it immediately (`python example.py`) to watch the compiler trace a dummy model, estimate its hardware costs, generate Zephyr C code, and verify the math.

```python
import torch
import torch.nn as nn
from oemgasqueeze import OemgaSqueeze

# 1. Define a supported 1D architecture
class Dummy1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=4 * 32, out_features=4)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return self.fc(self.flatten(x))

# 2. Generate synthetic data (Batch, Channels, Length)
# In production, replace this with your real biosensor dataset
input_shape = (1, 64)
example_input = torch.randn(1, *input_shape)
calibration_data = torch.randn(100, *input_shape) # Needed for INT8 scaling
X_test = torch.randn(50, *input_shape)
y_test = torch.randint(0, 4, (50,))

# 3. Initialize the compiler
model = Dummy1DCNN().eval()
compiler = OemgaSqueeze(
    model=model,
    example_input=example_input,
    calibration_data=calibration_data,
    output_dir="build_zephyr/app"
)

# 4. Compile to Zephyr C Code
print("Starting Compilation Pipeline...")
artifacts = compiler.compile(backend="oemga_native_int8")
print(f"Firmware generated in: {artifacts.output_root}")

# 5. Verify the generated C Model vs the PyTorch Model
print("\nStarting Verification...")
report = compiler.verify(X_test, y_test)
print(f"Prediction Agreement: {report['comparison']['prediction_agreement'] * 100:.2f}%")
```

## The Compilation Pipeline

When you call `.compile()`, OemgaSqueeze executes a 5-step pipeline, exposing transparency at every layer:

1.  **Frontend Normalization:** Uses `torch.fx` to trace the `nn.Module` and extract a normalized computational graph.
2.  **IR Lowering:** Maps PyTorch operations to a backend-independent Intermediate Representation (IR) with explicit shapes and datatypes.
3.  **Cost Analysis:** Statically calculates MACs, parameter bytes, and peak activation memory to estimate embedded deployment viability.
4.  **Backend Memory Planning:** Lowers the IR to an `oemga_native_int8` execution plan, scheduling static ping-pong memory buffers.
5.  **Code Generation:** Emits quantized weights, scales, and executable C/Zephyr code.

## Limitations (v1.0)

To maintain its strict, lightweight footprint, the current version supports a targeted subset of operations necessary for 1D signal processing:

  * **Operations:** `Conv1d` (groups=1), `Linear`, `MaxPool1d`, `ReLU`, `Reshape` (Flatten).
  * **Topology:** Currently supports sequential models with exactly one input tensor.
  * **Pooling Constraints:** `return_indices=True` and `ceil_mode=True` are not supported in `MaxPool1d`.
