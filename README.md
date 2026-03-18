<h1 align="center">OeMga.me Squeeze</h1>

<p align="center">
  <strong>A neural network compiler stack for ultra-low-power biosensor intelligence</strong>
</p>

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/>
  <img src="https://img.shields.io/badge/PyTorch-Frontend-ee4c2c.svg" alt="PyTorch Frontend"/>
  <img src="https://img.shields.io/badge/Codegen-Pure%20C-00599C.svg" alt="Pure C Codegen"/>
  <img src="https://img.shields.io/badge/Quantization-INT8-7A3E9D.svg" alt="INT8 Quantization"/>
  <img src="https://img.shields.io/badge/Target-Zephyr%20RTOS-black.svg" alt="Zephyr RTOS"/>
  <img src="https://img.shields.io/badge/Focus-ARM%20Cortex--M-0091BD.svg" alt="ARM Cortex-M"/>
</p>

**OemgaSqueeze** is a lightweight neural network compiler that translates PyTorch 1D CNN and Linear models into highly optimized, zero-dependency C code.

Built specifically for ultra-constrained edge AI, it removes the need for heavy third-party inference runtimes and instead statically compiles your model into a standalone execution pipeline. The generated code is designed for embedded deployment, with a strong focus on **Zephyr RTOS**, **wearable biosignal processing**, and **ARM Cortex-M-class microcontrollers**.

---

## Why OemgaSqueeze?

Traditional ML deployment stacks are often too heavy for deeply embedded systems. OemgaSqueeze is designed for the opposite end of the spectrum:

- ultra-low-power microcontrollers
- predictable memory usage
- zero dynamic allocation
- minimal firmware overhead
- efficient continuous inference on time-series signals

This makes it especially well suited for **real-time biosignal edge AI**, including applications such as **EMG-based gesture recognition** on wearable devices.

---

## Key Features

- **Zero-dependency C output**  
  Generates pure C source and header files such as `oemga_model.c` with no external inference libraries required.

- **No dynamic memory allocation**  
  No `malloc`, no runtime heap dependency, and no hidden framework overhead.

- **Static memory planning**  
  Uses ping-pong activation buffers and shared convolution scratchpads to keep SRAM usage bounded and predictable.

- **INT8 post-training quantization**  
  Profiles calibration data and quantizes weights and activations to INT8, with INT32 biases and per-channel requantization.

- **Zephyr-ready code generation**  
  Can emit an embedded application structure with files such as `main.c`, `CMakeLists.txt`, and `prj.conf` for direct Zephyr integration.

- **Pre-deployment hardware cost analysis**  
  Estimates flash and RAM consumption before deployment, including model weights, requantization parameters, double buffers, and scratch memory.

- **Host-to-target verification flow**  
  Generates a lightweight host-side C verifier to compare quantized C inference against the original PyTorch model using metrics such as macro F1 and logit difference reporting.

---

## Ideal Use Cases

OemgaSqueeze is optimized for low-power, continuous-monitoring edge applications.

### Biosignal Edge AI
Run continuous inference directly on sensor nodes for 1D time-series data such as:

- EMG
- ECG
- IMU-derived streams
- other wearable biosignals

### Open-Source Wearables
Designed for custom embedded AI platforms and open wearable systems where low latency and tight resource control matter.

### Strict RTOS Environments
Useful in systems where the following are critical:

- bounded memory usage
- deterministic execution
- low flash footprint
- low runtime overhead

---

## Installation

Clone the repository and install it in editable mode inside a virtual environment:

```bash
git clone https://github.com/yourusername/oemgasqueeze.git
cd oemgasqueeze
pip install -e .