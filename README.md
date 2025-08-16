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
    -DCMAKE_CXX_COMPILER=clang++ \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python)

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

sed -i '18 a\
set(LLVM_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/polygeist/llvm-project/build")\
set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm")\
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir")\
' ./CMakeLists.txt

mkdir build && cd build
cmake -G Ninja .. \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

ninja scalehls-opt scalehls-translate
```

- Build SAR-DSL

```bash
cd ../../..  # sar-dsl/
mkdir build && cd build
cmake -G Ninja .. \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

ninja
```

- Set env path

```bash
export PATH=$PWD/bin:$PATH && \
export PATH=$PWD/../externals/llvm-project/build/bin:$PATH && \
export PATH=$PWD/../externals/ScaleHLS-HIDA/build/bin:$PATH && \
export PYTHONPATH=$PWD/../externals/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH && \
export PYTHONPATH=$PWD/python/python_packages:$PYTHONPATH
```

- Test generate mlir

```bash
./test/test-gen-elem -o ../test/MLIR/test_gen_elem.mlir
./test/test-gen-fft -o ../test/MLIR/test_gen_fft.mlir
./test/test-shape-mismatch
```

- Lowering to Linalg

```bash
sar-opt ../test/MLIR/test_gen_elem.mlir --convert-sar-to-linalg \
    > ../test/MLIR/test_gen_elem_output.mlir
```

- Test LLVM output

```bash
mlir-opt ../test/MLIR/test_gen_elem_output.mlir \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-loops \
    --convert-scf-to-cf \
    --finalize-memref-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --convert-cf-to-llvm \
    --reconcile-unrealized-casts \
    | mlir-translate --mlir-to-llvmir > ../test/output.ll

clang -c ../test/output.ll -o ../test/output.o -Wno-override-module
clang -c ../test/ir_test.c -o ../test/ir_test.o
clang ../test/ir_test.o ../test/output.o -o ../test/ir_test
../test/ir_test
```

- Test ScaleHLS-HIDA

```bash
scalehls-opt ../test/test-scalehls-hida/affine_matmul.mlir \
    -hida-pytorch-pipeline="top-func=affine_matmul" \
    | scalehls-translate \
    -scalehls-emit-hlscpp -emit-vitis-directives \
    > ../test/emitHLS/hls_affine_matmul.cpp
```

- Test emit SAR to HLS

```bash
scalehls-opt ../test/MLIR/test_gen_elem_output.mlir \
    -hida-pytorch-pipeline="top-func=forward loop-tile-size=8 loop-unroll-factor=4" \
    | scalehls-translate \
    -scalehls-emit-hlscpp -emit-vitis-directives \
    > ../test/emitHLS/hls_output.cpp
```

- Test python frontend

```bash
python ../test/test_debug.py
python ../test/test_elem.py
python ../test/test_fft.py
```
