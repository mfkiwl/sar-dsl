# test/CMakeLists.txt

add_executable(sar-gen ir_gen.cpp)

target_link_libraries(sar-gen PRIVATE
    MLIRSARDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLinalgDialect
    MLIRParser
    MLIRPass
)

add_executable(test_shape_mismatch test_shape_mismatch.cpp)

target_link_libraries(test_shape_mismatch PRIVATE
    MLIRSARDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLinalgDialect
    MLIRParser
    MLIRPass
)

add_executable(test_gen_fft test_gen_fft.cpp)

target_link_libraries(test_gen_fft PRIVATE
    MLIRSARDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLinalgDialect
    MLIRParser
    MLIRPass
)
