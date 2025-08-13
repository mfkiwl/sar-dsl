# python/CMakeLists.txt

include(AddMLIRPython)

# Declare Python sources for the SAR dialect
declare_mlir_python_sources(SARPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sar"
    SOURCES
        __init__.py
        frontend.py
        _ods_common.py
)

# Create TableGen target for SAR ops Python bindings
set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/Dialect/SAR/IR/SAROps.td")
mlir_tablegen("_sar_ops_gen.py" -gen-python-op-bindings -bind-dialect=sar)
mlir_tablegen("_sar_enum_gen.py" -gen-python-enum-bindings)
add_public_tablegen_target(SARDialectPyOpsIncGen)
add_dependencies(SARDialectPyOpsIncGen MLIRSARDialectIncGen)

# Add generated files to sources
declare_mlir_python_sources(SARPythonSources.ops_gen
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
    ADD_TO_PARENT SARPythonSources
    SOURCES "_sar_ops_gen.py" "_sar_enum_gen.py"
)

# Declare the Python extension module for SAR dialect
declare_mlir_python_extension(SARPythonExtensions.Dialects.SAR
    MODULE_NAME _sarDialects
    ADD_TO_PARENT SARPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sar"
    SOURCES
        DialectSAR.cpp
    PRIVATE_LINK_LIBS
        LLVMSupport
    EMBED_CAPI_LINK_LIBS
        SARCAPI
)

# Aggregate and embed CAPI libs so that Python extensions resolve symbols at import time
add_mlir_python_common_capi_library(SARPythonCAPI
    INSTALL_COMPONENT SARPythonModules
    INSTALL_DESTINATION "python_packages/mlir/dialects/sar/_mlir_libs"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python_packages/mlir/dialects/sar/_mlir_libs"
    RELATIVE_INSTALL_ROOT "../../../../.."
    DECLARED_SOURCES SARPythonSources
)

# Create the Python modules
add_mlir_python_modules(SARPythonModules
    ROOT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/python_packages/mlir/dialects/sar"
    INSTALL_PREFIX "python_packages/mlir/dialects/sar"
    COMMON_CAPI_LINK_LIBS SARPythonCAPI
    DECLARED_SOURCES
        SARPythonSources
) 