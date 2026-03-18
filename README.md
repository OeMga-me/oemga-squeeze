<h1 align="center">OemgaSqueeze</h1>

<p align="center">
  <strong>A neural network compiler stack for ultra-low-power biosensor intelligence</strong>
</p>

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/>
  </a>
  <img src="https://img.shields.io/badge/Frontend-PyTorch-ee4c2c.svg" alt="PyTorch Frontend"/>
  <img src="https://img.shields.io/badge/Codegen-Pure%20C-00599C.svg" alt="Pure C Codegen"/>
  <img src="https://img.shields.io/badge/Target-Zephyr%20RTOS-black.svg" alt="Zephyr RTOS"/>
</p>

## Overview

**OemgaSqueeze** is a lightweight, purpose-built neural network compiler that translates PyTorch 1D CNN and Linear models into highly optimized, zero-dependency C code.

Designed for ultra-constrained edge AI, it removes the overhead of heavy third-party inference runtimes and instead statically compiles models into a standalone execution pipeline. The generated output is built to fit deeply embedded deployment settings, with a strong focus on **biosensor intelligence**, **predictable memory usage**, and **Zephyr RTOS integration**.

Rather than acting as a generic inference runtime, OemgaSqueeze follows a compile-first philosophy: take a trained model, lower it into a static embedded execution plan, and emit lean C code that is transparent, portable, and firmware-friendly.

## Why OemgaSqueeze?

Modern ML deployment tooling is often too heavy for tiny embedded systems. OemgaSqueeze is designed for the opposite regime:

- ultra-low-power microcontrollers
- predictable memory usage
- zero dynamic allocation
- low firmware overhead
- efficient continuous inference on time-series data

This makes it especially relevant for embedded systems where memory, latency, and deployment simplicity matter more than broad framework coverage.

## Key Features

- **Zero-dependency bare-metal C**  
  Generates pure C source and header files such as `oemga_model.c` with no external inference libraries required.

- **No dynamic memory allocation**  
  No `malloc`, no heap dependency, and no hidden runtime overhead.

- **Aggressive static memory planning**  
  Uses ping-pong activation buffers and shared scratchpads to keep SRAM usage bounded and predictable.

- **Zephyr RTOS ready**  
  Can generate an embedded application structure with files such as `main.c`, `CMakeLists.txt`, and `prj.conf` for straightforward Zephyr integration.

- **Pre-flight hardware cost analysis**  
  Provides a breakdown of estimated Flash and RAM requirements before deployment, including model parameters, double buffers, and scratch memory.

- **Host-to-target verification**  
  Generates a lightweight host-side verifier to compare generated C inference against the original PyTorch model using metrics such as macro F1 and logit difference reporting.

## Ideal Use Cases

OemgaSqueeze is optimized for low-power, continuous embedded inference workloads.

### Biosensor Intelligence
Designed for intelligent sensing systems where models must execute directly on-device under strict resource limits.

### Open Embedded Platforms
Useful for custom wearable, sensing, and edge devices that require a lean firmware-oriented ML deployment path.

### Strict RTOS Environments
A strong fit for systems where the following are critical:

- bounded memory usage
- deterministic execution
- minimal flash footprint
- transparent generated code
- straightforward integration into firmware stacks

## Installation

Clone the repository and install it in editable mode inside a virtual environment:

```bash
git clone https://github.com/yourusername/oemgasqueeze.git
cd oemgasqueeze
pip install -e .