# LeNet-5 Hardware Accelerator (VitisHLS)

> This project was built to explore High-Level Synthesis (HLS) as a bridge between high-level Machine Learning frameworks (Python/TensorFlow) and low-level Hardware deployment. It demonstrates a deep understanding of CNN architecture mapping to C++ and the application of Silicon-Mindset hardware directives to maximize DSP usage and throughput via Pipelining.

## Architecture Overview

This accelerator implements the classic LeNet-5 architecture for image classification (e.g., MNIST digits):
**Conv2D → MaxPool2D → Conv2D → MaxPool2D → Dense (Flatten) → Dense → Dense (Softmax)**

Key architectural features include:
- **Task-Level Pipelining:** Uses `#pragma HLS DATAFLOW` to overlap the execution of CNN layers. Instead of waiting for `Conv2D_0` to completely finish mapping an entire image, subsequent layers begin processing streamed chunks, vastly improving continuous throughput.
- **Loop Pipelining & Unrolling:** Uses `#pragma HLS PIPELINE` on the deepest nested Arithmetic loops (MAC operations) within Convolution and Dense layers. This fully unrolls the Dot-Product calculations, allowing multiple Multiplications and Accumulations to happen in a single clock cycle utilizing parallel DSP48 slices.
- **Fixed-Point Datatypes:** Migrated from standard floating-point `float` to `ap_fixed<32,16>` variables. This drastically reduces hardware area footprint (LUTs/FFs/BRAMs) while preserving enough precision for accurate CNN inference.

*Developed in C++ tailored for Xilinx Vivado HLS.*

## Code Structure

```text
Lenet-5-HLS/
├── python/                 // Python reference scripts mapping Keras/TensorFlow to plain numbers
├── files_from_Py2C/        // Core C++ files for HLS Synthesis
│   ├── CNN.cpp             // Top-level Inference Dataflow
│   ├── Conv.cpp            // Convolution operations & Line Buffering loops
│   ├── Pool.cpp            // Max Pooling reduction loops
│   ├── Dense.cpp           // Fully-connected & Softmax scoring
│   └── CNN_tb.cpp          // C++ Testbench to verify precision vs Python golden outputs
└── lenet_sys_vivado/       // Vivado GUI workspace & bitstream generation projects (if any)
```

## Running the Simulation & Synthesis

The project is designed to be compiled via **Xilinx Vivado HLS**.

To verify C-simulation locally (before Synthesizing to RTL):
1. Import all `.cpp` files in `files_from_Py2C` into a Vivado HLS project space.
2. Select `CNN.cpp` as the Top-Level Function.
3. Run **C Simulation** using `CNN_tb.cpp` as the testbench to verify fixed-point accuracy.
4. Run **C Synthesis** to compile the C++ behavior into an IP Core. Review the Synthesis Report to ensure the intervals and latencies meet expectations thanks to the PRAGMAs.


