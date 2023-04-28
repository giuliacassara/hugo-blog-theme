+++
date = "2023-04-28"
title = "Compiling a LLM to RISC-V via Torch MLIR and Google IREE"
slug = "llm-to-riscv"
readingtime = true
+++

In this article, we will explore how to compile a Large-Language Model like Bert and generate RISC-V vector extension code using the versatile MLIR ecosystem and LLVM target support.
We will walk you through the process of converting a high-level deep learning model into a lower-level representation suitable for RISC-V processors. Furthermore, we will discuss how to utilize LLVM's target support to generate efficient and optimized code that can be executed on these accelerators. 

>**Warning**: Although this may not be the most elegant solution, it is currently the only available option. The toolset employed is somewhat dispersed across different frameworks, which may lead to a higher likelihood of initial issues on your system. Nonetheless, with proper troubleshooting, you can overcome these challenges and successfully implement the process.


## Background

### RISC-V 
RISC-V is an open-source instruction set architecture (ISA) that is based on the Reduced Instruction Set Computer (RISC) principles. It has a modular design, which allows for extensibility and customization according to specific needs. RISC-V has gained significant attention for its flexibility, making it an attractive choice for various applications, including AI accelerators, embedded systems, and general-purpose processors.

The RISC-V Vector Extension (RVV) is an optional extension to the RISC-V ISA that introduces vector processing capabilities to the architecture. Vector processing allows for the simultaneous execution of operations on multiple data elements, which can lead to significant performance improvements, especially in compute-intensive tasks like AI and machine learning.

The Vector Extension adds new vector registers, instructions, and a configurable vector length, enabling developers to tailor the hardware to their specific requirements. This flexibility, combined with the inherent advantages of vector processing, makes RISC-V with Vector Extension a powerful platform for developing high-performance and energy-efficient AI accelerators and other specialized hardware.

## Software needed

### Google IREE 

Download Anaconda: [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)
Create a new conda environment and install IREE pip packages from pip.

```bash
$ conda create --name iree
$ pip install \
  iree-compiler \
  iree-runtime
```
**IREE** (Intermediate Representation Execution Environment) is a framework designed for efficient deep learning model execution on a variety of hardware platforms.
Now, go here and follow the instructions:
https://openxla.github.io/iree/getting-started/pytorch/



### LLVM 
```bash!
$ git clone https://github.com/llvm/llvm-project.git
$ git checkout tags/llvmorg-17.0.0
```
Clone the LLVM project repository and switch to a specific LLVM version (e.g., 17.0.0) by checking out the corresponding tag.
Make sure to uninstall previous clang installations!
```bash!
$ cmake -S llvm -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug \
                               -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld"
$ cd build && make -j$(nproc)
```
Configure the build using CMake, specifying the source directory (-S llvm), the build directory (-B build), the generator to use (Unix Makefiles), the build type (Debug), and the LLVM projects to enable (Clang, Clang Tools Extra, and LLD). 
```bash!
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/<yourpath>/llvm-project/build/lib
$ export PATH=$PATH:/<yourpath>/llvm-project/build/bin
```


## Compilation and dialect lowering


We start with this example. This code snippet demonstrates how to convert a BERT model into the STABLEHLO (MHLO) dialect using the torch_mlir library. In the MLIR ecosystem, dialects serve as domain-specific languages that cater to different levels of abstraction or target-specific representations. STABLEHLO (MHLO) is one such dialect, specifically designed for representing machine learning operations in a high-level and portable form.

```python 
import torch
import torch_mlir

from transformers import BertForMaskedLM

# Wrap the bert model to avoid multiple returns problem
class BertTinyWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained("prajjwal1/bert-tiny", return_dict=False)
    
    def forward(self, data):
        return self.bert(data)[0]

model = BertTinyWrapper()
model.eval()
data = torch.randint(30522, (2, 128))
out_stablehlo_mlir_path = "./bert_stablehlo.mlir"

module = torch_mlir.compile(model, data, output_type=torch_mlir.OutputType.STABLEHLO, use_tracing=True)
with open(out_stablehlo_mlir_path, "w", encoding="utf-8") as outf:
    outf.write(str(module))
    
```
The wrapper is needed to avoid issues with multiple return values that can arise when compiling the model using MLIR. The forward() method processes the input data and returns only the first output value from the BERT model. We instantiate the wrapped BERT model and set it to evaluation mode, create sample input data for the model, simulating tokenized text input for the BERT model.
Finally, we compile the model using the torch_mlir.compile() function with the specified input data, output type (STABLEHLO), and tracing enabled.


### IREE Compiler (LLVM Target)

We use the iree-compile command-line tool to compile the BERT-Tiny model StableHLO dialect into an IREE (Intermediate Representation Ecosystem for eXecution) binary. The output binary targets the RISC-V architecture with specific CPU features and configurations.

```bash!
iree-compile   --iree-hal-target-backends=llvm-cpu   --iree-llvm-target-triple=riscv64-unknown-elf   --iree-llvm-target-cpu=generic-rv64   --iree-llvm-target-abi=lp64d   --iree-llvm-target-cpu-features="+m,+a,+f,+d,+zvl1024b,+v"   --riscv-v-fixed-length-vector-lmul-max=1 --riscv-v-vector-bits-min=-1 --iree-input-type=stablehlo bert_tiny_stablehlo.mlir -o bert-tiny.vmfb
```

Let's break down some of the flags:

`--iree-hal-target-backends=llvm-cpu` specifies the target backend for the compilation, in this case, the LLVM backend for CPUs.

`--iree-llvm-target-triple=riscv64-unknown-elf` sets the target triple for the LLVM backend to RISC-V 64-bit architecture.

`--iree-llvm-target-cpu=generic-rv64` sets the target CPU for the LLVM backend to a generic RISC-V 64-bit processor.

`--iree-llvm-target-abi=lp64d` This flag specifies the target ABI (Application Binary Interface) for the LLVM backend. The lp64d ABI represents the ILP64 data model, where long and pointer types are 64-bit, and the d denotes support for double-precision floating-point operations.

`--iree-llvm-target-cpu-features="+m,+a,+f,+d,+zvl1024b,+v"` sets the specific CPU features for the target RISC-V processor, such as base integer instructions (+m), atomic instructions (+a), single-precision floating-point instructions (+f), double-precision floating-point instructions (+d), and vector extension (+v) with a maximum vector length of 1024 bits (+zvl1024b).

`--riscv-v-fixed-length-vector-lmul-max=1 --riscv-v-vector-bits-min=-1` flags set custom constraints for the RISC-V vector extension, such as maximum LMUL (length multiplier) and minimum vector bit width.

`--iree-input-type=mhlo` specifies that the input MLIR file uses the MHLO dialect.

`bert.mlir` is the input MLIR file containing the BERT-Tiny model in the StableHLO dialect.

`-o bert.vmfb`: sets the output file name for the compiled IREE binary.


In output, we will have a file called `bert.vmfb`. To access its contents, we can unpack it using the following command:

```bash!
$ unzip bert.vmfb
```

The file we are primarily interested in is the `.so` file, which contains the RISC-V executable kernel along with some glue logic for the IREE runtime environment. 

To inspect the contents of the ELF (Executable and Linkable Format) file, we can use the following command:


```bash!
$ llvm-objdump -d bert_linked_llvm_cpu_embedded_elf_riscv_64.so > bert.s
```

```bash!
$ bert_linked_llvm_cpu_embedded_elf_riscv_64.so:   file format elf64-littleriscv

Disassembly of section .text:

0000000000002640 <.text>:
    2640: 13 01 01 fb   addi    sp, sp, -80
    2644: 23 34 11 04   sd      ra, 72(sp)
    2648: 23 30 81 04   sd      s0, 64(sp)
    264c: 23 3c 91 02   sd      s1, 56(sp)
    2650: 23 38 21 03   sd      s2, 48(sp)
    2654: 23 34 31 03   sd      s3, 40(sp)
    2658: 23 30 41 03   sd      s4, 32(sp)
    265c: 23 3c 51 01   sd      s5, 24(sp)
    2660: 23 38 61 01   sd      s6, 16(sp)
    2664: 23 34 71 01   sd      s7, 8(sp)
    2668: 23 30 81 01   sd      s8, 0(sp)
    266c: 13 04 01 05   addi    s0, sp, 80
    2670: 03 b7 05 02   ld      a4, 32(a1)
    2674: 03 38 07 00   ld      a6, 0(a4)
    2678: 13 05 00 00   li      a0, 0
    267c: b7 95 0d 01   lui     a1, 4313
    2680: 9b 85 05 30   addiw   a1, a1, 768
    2684: b3 05 b8 00   add     a1, a6, a1
    2688: 83 36 87 00   ld      a3, 8(a4)
    
    ...
    
    2a70: a7 e5 02 02  	vse32.v	v11, (t0)
    2a74: 27 65 03 02  	vse32.v	v10, (t1)
    2a78: a7 e8 03 02  	vse32.v	v17, (t2)
    2a7c: 27 68 0e 02  	vse32.v	v16, (t3)
    2a80: a7 e7 0e 02  	vse32.v	v15, (t4)
    2a84: 27 67 0f 02  	vse32.v	v14, (t5)
    2a88: a7 e6 0f 02  	vse32.v	v13, (t6)
    2a8c: 27 66 09 02  	vse32.v	v12, (s2)
    
    ...
    
```


## References

1. RISC-V: The Free and Open RISC Instruction Set Architecture. 
2. Lattner, C., & Adve, V. (2004). LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation. LLVM Official Website
3. Lattner, Chris, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Zinenko. 2020. “MLIR: A Compiler Infrastructure for the End of Moore’s Law.” arXiv [cs.PL]. arXiv. http://arxiv.org/abs/2002.11054.
4. Google IREE https://github.com/openxla/iree 

