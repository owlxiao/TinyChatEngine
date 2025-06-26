#include "../matmul.h"

namespace matmul {

void MatmulOperator::mat_mul_accelerator_int8_fast_32unroll_over_column(const struct matmul_params *params) {}
void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll(const struct matmul_params *params) {}
void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(const struct matmul_params *params) {}
void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(const struct matmul_params *params) {}
void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(const struct matmul_params *params) {}
void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(const struct matmul_params *params) {
}
void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(const struct matmul_params *params) {}
void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(
    const struct matmul_params *params) {}

}  // namespace matmul