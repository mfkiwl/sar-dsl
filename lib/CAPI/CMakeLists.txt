add_mlir_public_c_api_library(SARCAPI
  Dialects.cpp
  
  DEPENDS
  MLIRSARDialectIncGen
  
  LINK_LIBS PUBLIC
  MLIRSARDialect
  MLIRSARToLinalg
) 