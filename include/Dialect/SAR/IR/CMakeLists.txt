# include/Dialect/SAR/IR/CMakeLists.txt

set(LLVM_TARGET_DEFINITIONS SAR.td)

mlir_tablegen(SARDialect.h.inc -gen-dialect-decls -dialect=sar)

mlir_tablegen(SARDialect.cpp.inc -gen-dialect-defs -dialect=sar)

mlir_tablegen(SARTypes.h.inc -gen-typedef-decls -dialect=sar)

mlir_tablegen(SARTypes.cpp.inc -gen-typedef-defs -dialect=sar)

mlir_tablegen(SAROps.h.inc -gen-op-decls -dialect=sar)

mlir_tablegen(SAROps.cpp.inc -gen-op-defs -dialect=sar)

add_public_tablegen_target(MLIRSARDialectIncGen)
