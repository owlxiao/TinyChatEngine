#include "sgemm.h"

#include <vx_print.h>
#include <vx_spawn.h>

void kernel(kernel_arg_t* __UNIFORM__ arg) {
    auto A = reinterpret_cast<float*>(arg->A_addr);
    auto B = reinterpret_cast<float*>(arg->B_addr);
    auto C = reinterpret_cast<float*>(arg->C_addr);
    auto k = arg->k;
    auto n = arg->n;
    auto m = arg->m;

    int col = blockIdx.x;
    int row = blockIdx.y;

    if (row >= m || col >= n) return;

    float sum(0);
    for (int e = 0; e < k; ++e) {
        sum += A[row * k + e] * B[col * k + e];
    }

    C[row * n + col] = sum;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel, arg);
}
