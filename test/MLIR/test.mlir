// test/MLIR/test.mlir

module @sar_module {
  func.func @sar_computation(%arg0: !sar.tensor<2x2xf32>, %arg1: !sar.tensor<2x2xf32>) -> !sar.tensor<2x2xf32> {
    %0 = "sar.const"() <{value = dense<1.0> : tensor<2x2xf32>}> : () -> !sar.tensor<2x2xf32>
    %1 = "sar.const"() <{value = dense<2.0> : tensor<2x2xf32>}> : () -> !sar.tensor<2x2xf32>
    %2 = "sar.add"(%arg0, %0) : (!sar.tensor<2x2xf32>, !sar.tensor<2x2xf32>) -> !sar.tensor<2x2xf32>
    %3 = "sar.mul"(%arg1, %1) : (!sar.tensor<2x2xf32>, !sar.tensor<2x2xf32>) -> !sar.tensor<2x2xf32>
    %4 = "sar.sub"(%2, %3) : (!sar.tensor<2x2xf32>, !sar.tensor<2x2xf32>) -> !sar.tensor<2x2xf32>
    return %4 : !sar.tensor<2x2xf32>
  }
}
