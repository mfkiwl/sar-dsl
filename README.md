# MLIR-based SAR Domain-Specific Accelerator Lang

The project is currently under development...

## Instructions

- List tree

```bash
tree -I "build" -I "llvm-project"
```

- Build LLVM & MLIR

```bash
cd externals/llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON

ninja

cd ../../..
```

- Compile

```bash
mkdir build && cd build
cmake -G Ninja ..
ninja
```

- Set path

```bash
export PATH=$PWD/bin:$PATH
export PATH=$PWD/../externals/llvm-project/build/bin:$PATH
```

- Generate mlir

```bash
./test/sar-gen > ../test/MLIR/test.mlir
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

- Clean

```bash
cd .. && rm -rf build && mkdir build && cd build && clear
```
