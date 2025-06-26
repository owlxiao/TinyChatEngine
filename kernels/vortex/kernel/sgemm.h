#pragma once

#include <cstdint>

typedef struct {
    uint32_t grid_dim[2];
    uint32_t k;
    uint32_t n;
    uint32_t m;
    uint64_t A_addr;
    uint64_t B_addr;
    uint64_t C_addr;
} kernel_arg_t;