// test/ir_test.c

#include <stdio.h>
#include <stdlib.h>

// Structure matching LLVM function signature
typedef struct {
    float *allocated;  // Pointer to allocated memory
    float *aligned;    // Pointer to aligned memory
    long offset;       // Offset for indexed access
    long sizes[2];     // Dimension sizes
    long strides[2];   // Memory strides
} MemRef;

// Declare LLVM function
extern MemRef sar_computation(
    float* a_alloc, float* a_align, long a_offset, 
    long a_size0, long a_size1, long a_stride0, long a_stride1,
    float* b_alloc, float* b_align, long b_offset, 
    long b_size0, long b_size1, long b_stride0, long b_stride1
);

int main() {
    // Input data (row-major layout)
    float in1_data[4] = {1.0, 2.0, 3.0, 4.0};  // Represents [[1,2],[3,4]]
    float in2_data[4] = {5.0, 6.0, 7.0, 8.0};  // Represents [[5,6],[7,8]]

    // Create input memref descriptors
    MemRef input1 = {
        .allocated = in1_data,
        .aligned = in1_data,
        .offset = 0,
        .sizes = {2, 2},     // 2x2 matrix
        .strides = {2, 1}     // Row-major layout
    };

    MemRef input2 = {
        .allocated = in2_data,
        .aligned = in2_data,
        .offset = 0,
        .sizes = {2, 2},     // 2x2 matrix
        .strides = {2, 1}     // Row-major layout
    };

    // Call LLVM function
    MemRef result = sar_computation(
        input1.allocated, input1.aligned, input1.offset,
        input1.sizes[0], input1.sizes[1],
        input1.strides[0], input1.strides[1],

        input2.allocated, input2.aligned, input2.offset,
        input2.sizes[0], input2.sizes[1],
        input2.strides[0], input2.strides[1]
    );

    // Check if result pointer is valid
    if (result.aligned == NULL) {
        printf("Error: Result pointer is NULL!\n");
        return 1;
    }

    // Print result matrix
    printf("Result:\n");
    for (int i = 0; i < 2; i++) {
        printf("[ ");
        for (int j = 0; j < 2; j++) {
            // Calculate element position using strides
            long index = i * result.strides[0] + j * result.strides[1];
            printf("%.1f ", result.aligned[index]);
        }
        printf("]\n");
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
// Result:
// [ -8.0 -9.0 ]
// [ -10.0 -11.0 ]
