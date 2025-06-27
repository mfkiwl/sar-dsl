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
