# tools/sar-opt/CMakeLists.txt

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
    SARTransforms
    MLIRSARDialect
    MLIRSARToLinalg
)

add_mlir_tool(sar-opt
    sar-opt.cpp

    DEPENDS
    ${LIBS}
)

target_link_libraries(sar-opt PRIVATE ${LIBS})

llvm_update_compile_flags(sar-opt)

mlir_check_all_link_libraries(sar-opt)

export_executable_symbols_for_plugins(sar-opt)
