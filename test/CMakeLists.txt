# test/CMakeLists.txt

add_executable(test-gen-elem test_gen_elem.cpp)

target_link_libraries(test-gen-elem PRIVATE
    MLIRIR
    MLIRSARDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLinalgDialect
    MLIRParser
    MLIRPass
)

add_executable(test-shape-mismatch test_shape_mismatch.cpp)

target_link_libraries(test-shape-mismatch PRIVATE
    MLIRIR
    MLIRSARDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLinalgDialect
    MLIRParser
    MLIRPass
)

add_executable(test-gen-fft test_gen_fft.cpp)

target_link_libraries(test-gen-fft PRIVATE
    MLIRIR
    MLIRSARDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLinalgDialect
    MLIRParser
    MLIRPass
)
