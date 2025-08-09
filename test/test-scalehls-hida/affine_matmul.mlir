func.func @affine_matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  
  // Matrix multiplication using affine loops
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      affine.for %k = 0 to 4 {
        %a_val = memref.load %A[%i, %k] : memref<4x4xf32>
        %b_val = memref.load %B[%k, %j] : memref<4x4xf32>
        %c_val = memref.load %C[%i, %j] : memref<4x4xf32>
        
        %prod = arith.mulf %a_val, %b_val : f32
        %sum = arith.addf %c_val, %prod : f32
        
        memref.store %sum, %C[%i, %j] : memref<4x4xf32>
      }
    }
  }
  
  return
}
