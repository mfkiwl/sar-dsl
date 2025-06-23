# lib/Dialect/SAR/IR/CMakeLists.txt

add_mlir_dialect_library(MLIRSARDialect
    SARDialect.cpp
    SARTypes.cpp
    SAROps.cpp

    DEPENDS
    SARDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
)
