# MLIR-based SAR Domain-Specific Accelerator Lang

The project is currently under development...

## Getting Started

### Instructions

- Clone repository

```bash
git clone https://github.com/zeroherolin/sar-dsl.git
cd sar-dsl && git submodule update --init --recursive
```

- Build MLIR

```bash
cd externals/llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

ninja
```

- Build ScaleHLS

Duplicated MLIR is built here cuz' ScaleHLS depends on the old version.

```bash
cd ../../ScaleHLS-HIDA/polygeist/llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

ninja

cd ../../..
# edit externals/ScaleHLS-HIDA/CMakeLists.txt and add the following: 
set(LLVM_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/polygeist/llvm-project/build")
set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm")
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir")

mkdir build && cd build
cmake -G Ninja .. \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

ninja scalehls-opt scalehls-translate
```

- Build SAR-DSL

```bash
cd ../../.. # sar-dsl/
mkdir build && cd build
cmake -G Ninja ..
ninja
```

- Set env path

```bash
export PATH=$PWD/bin:$PATH
export PATH=$PWD/../externals/llvm-project/build/bin:$PATH
export PATH=$PWD/../externals/ScaleHLS-HIDA/build/bin:$PATH
```

- Generate mlir

```bash
./test/sar-gen -o ../test/MLIR/test.mlir
```

- Pass & Lowering

```bash
sar-opt ../test/MLIR/test.mlir --convert-sar-to-linalg > ../test/MLIR/output.mlir

mlir-opt ../test/MLIR/output.mlir \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops \
    --convert-scf-to-cf \
    --finalize-memref-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --convert-cf-to-llvm \
    --reconcile-unrealized-casts \
    | mlir-translate --mlir-to-llvmir > ../test/MLIR/output.ll
```

- Test output

```bash
clang -c ../test/MLIR/output.ll -o ../test/output.o -Wno-override-module
clang -c ../test/ir_test.c -o ../test/ir_test.o
clang ../test/ir_test.o ../test/output.o -o ../test/ir_test
../test/ir_test
```

- Test emitHLS

```bash
scalehls-opt ../test/MLIR/output.mlir \
    -hida-pytorch-pipeline="top-func=forward loop-tile-size=8 loop-unroll-factor=4" \
    | scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives > ../test/emitHLS.cpp
```
