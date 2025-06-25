# tools/CMakeLists.txt

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}

    MLIRAffineAnalysis
    MLIRAnalysis
    MLIRCastInterfaces
    MLIRDialect
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRSupport
    MLIRIR

    MLIRToLLVMIRTranslationRegistration
    MLIRSARDialect
    MLIRSARToLinalg
)

add_mlir_tool(sar-opt sar-opt.cpp)

target_link_libraries(sar-opt
    PRIVATE
    ${LIBS}
)
