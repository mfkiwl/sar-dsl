add_mlir_dialect_library(MLIRSARDialect
  SARDialect.cpp
  SARTypes.cpp
  SAROps.cpp

  DEPENDS
  MLIRSARDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAsmParser
  MLIRTensorDialect
)
