#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../matmul.h"
#include "./kernel/sgemm.h"
#include "vortex.h"

namespace matmul {

// ----------------------------------------------------------------------------
// Vortex

#define RT_CHECK(_expr)                                          \
    do {                                                         \
        int _ret = _expr;                                        \
        if (0 == _ret) break;                                    \
        printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
        cleanup();                                               \
        exit(-1);                                                \
    } while (false)

using vx_addr_h = uint64_t;

static const char *vortex_kernel = "../kernels/vortex/kernel/kernel.vxbin";
static vx_buffer_h vortex_buffer = nullptr;
static vx_buffer_h vortex_args_buffer = nullptr;

static vx_device_h device = nullptr;

static void cleanup() {
    // Free buffers
    if (vortex_buffer) {
        RT_CHECK(vx_mem_free(vortex_buffer));
        vortex_buffer = nullptr;
    }
    if (vortex_args_buffer) {
        RT_CHECK(vx_mem_free(vortex_args_buffer));
        vortex_args_buffer = nullptr;
    }

    // Close Vortex device connection
    if (device) {
        RT_CHECK(vx_dev_close(device));
        device = nullptr;
    }
}

__attribute__((constructor)) static void __vortex_module_ctor() {
    // Open Vortex device connection
    RT_CHECK(vx_dev_open(&device));

    // Upload kernels
    RT_CHECK(vx_upload_kernel_file(device, vortex_kernel, &vortex_buffer));
}

__attribute__((destructor)) static void __vortex_module_dtor() { cleanup(); }

// ----------------------------------------------------------------------------
// TinyChatEngine Kernel

void fp32_ref_matmul(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * k + kk] * data_B[j * k + kk];
            }
            acc = acc;
            data_C[i * n + j] = acc;
            // std::cout << "@ref C[" << i << "][" << j << "] = " << acc << std::endl;  // Debug output
        }
    }
}

bool float_equal(float a, float b, float epsilon) { return fabs(a - b) < epsilon; }

void MatmulOperator::mat_mul_accelerator_transposed_fastover_column(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    // std::vector<float> vortex_c(C->length(), 0.0f);
    int m = A->row, n = B->column, k = A->column;
    // print matrix A, B, C dimensions
    // std::cout << "Matrix A: " << A->row << "x" << A->column << std::endl;
    // std::cout << "Matrix B: " << B->row << "x" << B->column << std::endl;
    // std::cout << "Matrix C: " << C->row << "x" << C->column << "\n" << std::endl;

    // Vortex kernel parameters
    vx_buffer_h A_buffer = nullptr;
    vx_buffer_h B_buffer = nullptr;
    vx_buffer_h C_buffer = nullptr;
    kernel_arg_t args;

    // allocate buffers for A, B, C matrix
    RT_CHECK(vx_mem_alloc(device, A->length() * sizeof(float), VX_MEM_READ, &A_buffer));
    RT_CHECK(vx_mem_address(A_buffer, &args.A_addr));
    RT_CHECK(vx_mem_alloc(device, B->length() * sizeof(float), VX_MEM_READ, &B_buffer));
    RT_CHECK(vx_mem_address(B_buffer, &args.B_addr));
    RT_CHECK(vx_mem_alloc(device, C->length() * sizeof(float), VX_MEM_WRITE, &C_buffer));
    RT_CHECK(vx_mem_address(C_buffer, &args.C_addr));

    // Upload matrix A buffer
    RT_CHECK(vx_copy_to_dev(A_buffer, data_A, 0, A->length() * sizeof(float)));
    RT_CHECK(vx_copy_to_dev(B_buffer, data_B, 0, B->length() * sizeof(float)));

    // Upload kernel arguments
    args.grid_dim[0] = n;
    args.grid_dim[1] = m;
    args.k = k;
    args.n = n;
    args.m = m;
    RT_CHECK(vx_upload_bytes(device, &args, sizeof(args), &vortex_args_buffer));

    // Start the kernel
    RT_CHECK(vx_start(device, vortex_buffer, vortex_args_buffer));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    // Download result matrix C
    RT_CHECK(vx_copy_from_dev(data_C, C_buffer, 0, C->row * C->column * sizeof(float)));

    // Free the buffers
    RT_CHECK(vx_mem_free(A_buffer));
    RT_CHECK(vx_mem_free(B_buffer));
    RT_CHECK(vx_mem_free(C_buffer));

    // fp32_ref_matmul(params);  // Call reference implementation for verification
    // for (int i = 0; i < C->row * C->column; i++) {
    //     if (!float_equal(vortex_c[i], data_C[i], 0.0001)) {  // Check for NaN
    //         std::cout << "Mismatch at index " << i << ": "
    //                   << "Vortex result = " << vortex_c[i] << ", "
    //                   << "Reference result = " << data_C[i] << std::endl;
    //         throw std::runtime_error("Vortex matmul result does not match reference implementation.");
    //     }
    // }
}

void fp32_ref_matmul_bias(const struct matmul_params *params) {
    std::cout << "Using reference implementation for FP32 matmul with bias." << std::endl;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *bias = params->bias.data_ptr;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * k + kk] * data_B[j * k + kk];
            }
            acc = acc + bias[j];
            data_C[i * n + j] = acc;
        }
    }
}

void MatmulOperator::mat_mul_accelerator_transposed_fastover_column_bias(const struct matmul_params *params) {
    fp32_ref_matmul_bias(params);
}

};  // namespace matmul