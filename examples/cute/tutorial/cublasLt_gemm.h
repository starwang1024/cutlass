#include <cublasLt.h>

#define check(call)                                        \
  do {                                                     \
    auto err = call;                                       \
    if (err != CUBLAS_STATUS_SUCCESS) {                    \
      printf("err = %d, str = %s, line = %d, %s\n", err,   \
             cublasGetStatusString(err), __LINE__, #call); \
      exit(0);                                             \
    }                                                      \
  } while (0)

template <typename T>
struct ComputeTypeTraits {
  static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_16F;
  static constexpr cudaDataType_t kScaleType = CUDA_R_16F;
};

template <>
struct ComputeTypeTraits<float> {
  static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32F;
  static constexpr cudaDataType_t kScaleType = CUDA_R_32F;
};

// 添加int8_t的特化，计算结果为int32_t
template <>
struct ComputeTypeTraits<int8_t> {
  static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32I;
  static constexpr cudaDataType_t kScaleType = CUDA_R_32I;
};

// 添加int32_t的特化
template <>
struct ComputeTypeTraits<int32_t> {
  static constexpr cublasComputeType_t kComputeType = CUBLAS_COMPUTE_32I;
  static constexpr cudaDataType_t kScaleType = CUDA_R_32I;
};

template <typename T, typename ComputeType>
struct CublasLtGemm {
  cublasLtHandle_t handle_;

  cublasLtMatrixLayout_t a_desc_;
  cublasLtMatrixLayout_t b_desc_;
  cublasLtMatrixLayout_t c_desc_;

  cublasLtMatmulDesc_t matmul_desc_;

  cublasLtMatmulPreference_t preference_;

  static constexpr int kAlgoMaxNum = 1024;
  cublasLtMatmulHeuristicResult_t algos_[kAlgoMaxNum];
  int ret_algo_num_;

  ComputeType alpha_;
  ComputeType beta_;
  static constexpr cublasComputeType_t kComputeType =
      ComputeTypeTraits<ComputeType>::kComputeType;
  static constexpr cudaDataType_t kScaleType =
      ComputeTypeTraits<ComputeType>::kScaleType;

  void *workspace_;
  int workspace_size_;

  const void *a_;
  const void *b_;
  void *c_;

  void init(ComputeType *c, const T *a, const T *b, int m, int n, int k);
  bool run();
};

template <typename T, typename ComputeType>
bool CublasLtGemm<T, ComputeType>::run() {
  auto algo = algos_[0];

  check(cublasLtMatmul(handle_, matmul_desc_, &alpha_, a_, a_desc_, b_,
                          b_desc_, &beta_, c_, c_desc_, c_, c_desc_,
                          &(algo.algo), workspace_, workspace_size_, 0));

  return true;
}

template <typename T, typename ComputeType>
void CublasLtGemm<T, ComputeType>::init(ComputeType *c, const T *a, const T *b, int m,
                                        int n, int k) {
  auto version = cublasLtGetVersion();
  printf("cublasLt version: %zu\n", version);

  check(cublasLtCreate(&handle_));

  // cublasLtLoggerSetLevel(5);

  int batch = 1;
  int64_t a_stride = m * k;
  int64_t b_stride = n * k;
  int64_t c_stride = m * n;
  
  // 设置矩阵操作类型
  // 对于行主序的A，不需要转置
  // 对于列主序的B，不需要转置
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  // 设置数据类型和布局
  // 对于int8_t特化
  if constexpr (std::is_same<T, int8_t>::value) {
    // A: 行主序 (ROW_MAJOR), int8
    check(cublasLtMatrixLayoutCreate(&a_desc_, CUDA_R_8I, m, k, k));
    // 设置为行主序
    cublasLtOrder_t order_row = CUBLASLT_ORDER_ROW;
    check(cublasLtMatrixLayoutSetAttribute(
        a_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
    
    // B: 列主序 (COLUMN_MAJOR), int8
    check(cublasLtMatrixLayoutCreate(&b_desc_, CUDA_R_8I, k, n, k));
    // 列主序是默认的，不需要特别设置
    
    // C/D: 行主序 (ROW_MAJOR), int32
    check(cublasLtMatrixLayoutCreate(&c_desc_, CUDA_R_32I, m, n, n));
    // 设置为行主序
    check(cublasLtMatrixLayoutSetAttribute(
        c_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
  } else {
    // 原始的fp16实现
    check(cublasLtMatrixLayoutCreate(&a_desc_, CUDA_R_16F, k, m, k));
    check(cublasLtMatrixLayoutCreate(&b_desc_, CUDA_R_16F, k, n, k));
    check(cublasLtMatrixLayoutCreate(&c_desc_, CUDA_R_16F, m, n, m));
  }

  check(cublasLtMatrixLayoutSetAttribute(
      a_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
  check(cublasLtMatrixLayoutSetAttribute(
      b_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
  check(cublasLtMatrixLayoutSetAttribute(
      c_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));

  check(cublasLtMatrixLayoutSetAttribute(
      a_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride,
      sizeof(a_stride)));
  check(cublasLtMatrixLayoutSetAttribute(
      b_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride,
      sizeof(b_stride)));
  check(cublasLtMatrixLayoutSetAttribute(
      c_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride,
      sizeof(c_stride)));

  check(cublasLtMatmulDescCreate(&matmul_desc_, kComputeType, kScaleType));
  check(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  check(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  // 设置alpha和beta
  if constexpr (std::is_same<ComputeType, int32_t>::value) {
    alpha_ = 1;
    beta_ = 0;  // 设置为1，实现D = A*B + C
  } else {
    alpha_ = 1.f;
    beta_ = 0.f;
  }
  
  workspace_ = nullptr;
  workspace_size_ = 0;

  // 创建首选项对象
  check(cublasLtMatmulPreferenceCreate(&preference_));
  
  // 设置工作区大小限制
  size_t workspaceSize = 1024 * 1024 * 4;  // 4MB
  check(cublasLtMatmulPreferenceSetAttribute(
      preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceSize, sizeof(workspaceSize)
  ));

  // 查询最佳算法
  check(cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc_, a_desc_, b_desc_,
                                 c_desc_, c_desc_, preference_, kAlgoMaxNum,
                                 algos_, &ret_algo_num_));

  if (ret_algo_num_ == 0) {
    printf("无法找到合适的算法！\n");
    exit(EXIT_FAILURE);
  }

  a_ = a;
  b_ = b;
  c_ = c;
}
 