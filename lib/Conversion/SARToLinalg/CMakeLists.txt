add_mlir_conversion_library(MLIRSARToLinalg
  SARToLinalg.cpp
  SARToLinalgPass.cpp

  DEPENDS
  MLIRSARConversionPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAsmParser
  MLIRPass
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRSARDialect
)
