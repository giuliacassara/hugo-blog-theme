+++
date = "2023-04-07"
title = "How to generate RISC-V vector extension assembly from Tensorflow"
slug = "riscv-from-tf"
readingtime = true
+++

In this tutorial, we will see how to generate RISC-V vector extension assembly from Tensorflow via XLA.

## What is XLA?

In TensorFlow, graph execution means that tensor computations are executed as a TensorFlow graph, sometimes referred to as a `tf.Graph` or simply a “graph”. Computations are described using a data-flow-like model, and the computations can be mapped onto different hardware like CPUs and GPUs.

**XLA** (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes.

XLA compiler provides the following advantages:
* Fused pipeline operations to reduce memory overhead
* Memory usage analysis to eliminate intermediate buffer usage
* Fusing of operations/kernels to form a low-level op to match the performance of custom-tuned low-level Operations

Tensorflow, with `tf.function` support graph mode, generates an intermediate graph object that is later parsed and optimized by XLA, producing HLO, and from there, we can emit the LLVM/IR code.

## Getting started with XLA
### Test Workload

By default, in Tensorflow 2.x, the code runs in eager mode. To run in graph mode, we use a `@tf.fuction` decorator such that the whole function will be compiled, optimized, and run as a single computational graph.

```python
@tf.function(experimental_compile=True)
def myworkload(a,b):
    return tf.add(a,b)
```

### Inspecting compiled programs
You can inspect compiled program in text format and html format where you can see your computational graph visually.
 by setting a bunch of environment variables before launching the workload:
 ```bash
 XLA_FLAGS=" - xla_dump_to=/tmp/generated" TF_XLA_FLAGS=" - tf_xla_auto_jit=2" my/tensorflow/program
 ```

 ## Generate RISC-V vector extension assembly
Select your compiled workload. Inside the folder specified during compilation time, you will find files with .ll extensions, which are LLVM IR files. We are interested in modules that end with `.ir-with-opt.ll`.

This command will help you navigate all the attributes needed to generate RISC-V assembly code from LLVM-IR:
```bash
$ llc -march=riscv64 -mattr=help
```

To produce the RISC-V vector code:
```bash
find ./ -type f -name "*ir-with-opt.ll" -exec llc "{}" -march=riscv64 -mattr=+m,+v,+zve32x,+zvl1024b - riscv \;
```

Where:
* `+m` : support multiplication
* `+v`: vector extension
* `+zve32x`: it’s one of the five standard extensions defined to provide varying degrees of vector support for embedded processors
* `+zvl1024b`: Specify that the minimum VLEN is 1024.

## References
* [TensorFlow XLA](https://www.tensorflow.org/xla)
* [Intro to Graphs](https://www.tensorflow.org/guide/intro_to_graphs)
* [Technical detail about Loop fusion, loop unrolling](http://akira.ruc.dk/~keld/teaching/IPDC_f10/Slides/pdf4x/4_Performance.4x.pdf)
* [Example of assembly code generation from C](https://www.luffca.com/2022/06/riscv-vector-vicuna-simulator/)
* [RISC-V Vector Specs](https://github.com/riscv/riscv-v-spec/releases/tag/v1.0)