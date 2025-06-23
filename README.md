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
    -DLLVM_ENABLE_PROJECTS=mlir \
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

- Run

```bash
./test/sar_test
```

- Clean

```bash
cd .. && rm -rf build && mkdir build && cd build && clear
```
