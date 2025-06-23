# lib/Conversion/SARToLinalg/CMakeLists.txt

add_mlir_conversion_library(MLIRSARToLinalg
    SARToLinalg.cpp
    SARToLinalgPass.cpp

    DEPENDS
    MLIRSARConversionPassesIncGen

    LINK_LIBS PUBLIC
    MLIRLinalgDialect
    MLIRSARDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRPass
    MLIRTransforms
)