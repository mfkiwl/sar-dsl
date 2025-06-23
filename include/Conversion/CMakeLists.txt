# include/Conversion/CMakeLists.txt

set(LLVM_TARGET_DEFINITIONS Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls -name SARConversion)

add_public_tablegen_target(MLIRSARConversionPassesIncGen)

add_dependencies(mlir-headers MLIRSARConversionPassesIncGen)

