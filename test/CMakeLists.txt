# test/CMakeLists.txt

add_executable(sar_test test.cpp)

target_link_libraries(sar_test PRIVATE
    MLIRSARDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLinalgDialect
    MLIRParser
    MLIRPass
)
