add_executable(sar_test test.cpp)

target_link_libraries(sar_test PRIVATE
    MLIRSARDialect
    MLIRGPUDialect
    MLIRFuncDialect
)
