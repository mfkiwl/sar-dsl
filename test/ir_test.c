// test/ir_test.c

#include <stdio.h>
#include <stdlib.h>

// Structure for 3D memref descriptor
typedef struct {
    float *allocated;  // Pointer to allocated memory
    float *aligned;    // Pointer to aligned memory
    long offset;       // Offset for indexed access
    long sizes[3];     // Dimension sizes [depth, height, width]
    long strides[3];   // Memory strides [depth_stride, height_stride, width_stride]
} MemRef3D;

// Declare LLVM function for 3D tensors
extern MemRef3D sar_computation(
    float* a_alloc, float* a_align, long a_offset, 
    long a_size0, long a_size1, long a_size2, 
    long a_stride0, long a_stride1, long a_stride2,
    float* b_alloc, float* b_align, long b_offset, 
    long b_size0, long b_size1, long b_size2,
    long b_stride0, long b_stride1, long b_stride2
);

int main() {
    // Input data (row-major layout for 3x3x3 tensors)
    float in1_data[27], in2_data[27];
    
    // Initialize with sample values (1.0 and 2.0 throughout)
    for (int i = 0; i < 27; i++) {
        in1_data[i] = 1.0f;
        in2_data[i] = 2.0f;
    }

    // Create input memref descriptors for 3D tensors
    MemRef3D input1 = {
        .allocated = in1_data,
        .aligned = in1_data,
        .offset = 0,
        .sizes = {3, 3, 3},   // 3x3x3 tensor
        .strides = {9, 3, 1}  // Row-major layout
    };
    
    MemRef3D input2 = {
        .allocated = in2_data,
        .aligned = in2_data,
        .offset = 0,
        .sizes = {3, 3, 3},   // 3x3x3 tensor
        .strides = {9, 3, 1}  // Row-major layout
    };
    
    // Call LLVM function
    MemRef3D result = sar_computation(
        input1.allocated, input1.aligned, input1.offset,
        input1.sizes[0], input1.sizes[1], input1.sizes[2],
        input1.strides[0], input1.strides[1], input1.strides[2],
        
        input2.allocated, input2.aligned, input2.offset,
        input2.sizes[0], input2.sizes[1], input2.sizes[2],
        input2.strides[0], input2.strides[1], input2.strides[2]
    );
    
    // Check if result pointer is valid
    if (result.aligned == NULL) {
        printf("Error: Result pointer is NULL!\n");
        return 1;
    }
    
    // Print result tensor
    printf("Result tensor (3x3x3):\n");
    for (int d = 0; d < result.sizes[0]; d++) {
        printf("Depth %d:\n", d);
        for (int h = 0; h < result.sizes[1]; h++) {
            printf("  [ ");
            for (int w = 0; w < result.sizes[2]; w++) {
                // Calculate element position using strides
                long index = d * result.strides[0] + 
                             h * result.strides[1] + 
                             w * result.strides[2];
                printf("%.1f ", result.aligned[index]);
            }
            printf("]\n");
        }
    }
    
    // Free heap-allocated memory (if any)
    if (result.allocated != in1_data && result.allocated != in2_data) {
        free(result.allocated);
    } else {
        printf("Note: Result uses input memory, no free needed\n");
    }
    
    return 0;
}

// Expected output:
// Result tensor (3x3x3):
// Depth 0:
//   [ -2.0 -2.0 -2.0 ]
//   [ -2.0 -2.0 -2.0 ]
//   [ -2.0 -2.0 -2.0 ]
// Depth 1:
//   [ -2.0 -2.0 -2.0 ]
//   [ -2.0 -2.0 -2.0 ]
//   [ -2.0 -2.0 -2.0 ]
// Depth 2:
//   [ -2.0 -2.0 -2.0 ]
//   [ -2.0 -2.0 -2.0 ]
//   [ -2.0 -2.0 -2.0 ]
