#include "../matmul.h"

namespace matmul {

void MatmulOperator::mat_mul_accelerator_int4_fast(const struct matmul_params *params) {}
void MatmulOperator::mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params *params) {}
void MatmulOperator::mat_mul_accelerator_int8_int4_fast_no_offset(struct matmul_params *params) {}
void MatmulOperator::gemv_accelerator_int8_int4_fast_no_offset(struct matmul_params *params) {}
void MatmulOperator::gemm_accelerator_int8_int4_fast_no_offset(struct matmul_params *params) {}
void MatmulOperator::gemm_accelerator_int8_int4_fast_no_offset_v2(struct matmul_params *params) {}
void MatmulOperator::cblas_gemm_accelerator_no_offset(struct matmul_params *params) {}

}  // namespace matmul