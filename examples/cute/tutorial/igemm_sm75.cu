// nvcc -std=c++17 -O3 -DNDEBUG igemm_sm75.cu -Icutlass/include -Icutlass/tools/util/include -Icutlass/examples/common -lcudart -lcublas -lcublasLt -arch=sm_75
// Problem size: 64x40960x1024
// CUTE_GEMM:     [10723.4]GFlop/s [ 104.9]GB/s  (0.5007)ms
// CUTE_GEMM:     [12901.8]GFlop/s [ 126.2]GB/s  (0.4161)ms
// CUTE_GEMM:     [12907.7]GFlop/s [ 126.2]GB/s  (0.4159)ms
// CUTE_GEMM:     [14558.4]GFlop/s [ 142.3]GB/s  (0.3688)ms
// CUTE_GEMM:     [16167.5]GFlop/s [ 158.1]GB/s  (0.3321)ms
// CUTE_GEMM:     [18525.6]GFlop/s [ 181.1]GB/s  (0.2898)ms
// CUTE_GEMM:     [19115.0]GFlop/s [ 186.9]GB/s  (0.2809)ms
// CUTE_GEMM:     [20064.9]GFlop/s [ 196.2]GB/s  (0.2676)ms
// CUBLAS_GEMM:   [19984.0]GFlop/s [ 195.4]GB/s  (0.2687)ms

#include <iostream>  
#include <cutlass/cutlass.h>  
#include <cutlass/numeric_types.h>  
#include <cutlass/gemm/gemm.h>  
#include <cutlass/gemm/dispatch_policy.hpp>  
#include <cutlass/gemm/collective/collective_builder.hpp>  
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp>
#include <cutlass/gemm/kernel/gemm_universal.h>  
#include <cutlass/gemm/device/gemm_universal_adapter.h>  
#include <cutlass/layout/matrix.h>  
#include <cutlass/util/host_tensor.h>  
#include <cutlass/util/reference/host/tensor_fill.h>  
#include <cutlass/util/reference/host/gemm.h>  
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include "cutlass/util/GPU_Clock.hpp"
#include "cublasLt_gemm.h"

using namespace cute;  
using namespace cutlass;  
using namespace cutlass::gemm;  

int main(int argc, char** argv) {

    int m = 512;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 512;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 512;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    static constexpr int kTileM = 32;
    static constexpr int kTileN = 128;
    static constexpr int kTileK = 128;

    using DispatchPolicy = MainloopSm70TwoStageUnpredicated;
    using ElementA = int8_t;  
    using ElementB = int8_t;  
    using ElementC = int32_t;  
    using ElementAccumulator = int32_t;

    using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;  
    using LayoutA = cutlass::layout::RowMajor;  
    using LayoutB = cutlass::layout::ColumnMajor;  
    using LayoutC = cutlass::layout::RowMajor;  

    using mma_op = SM75_8x8x16_S32S8S8S32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 1;
    static constexpr int kMmaEURepeatN = 4;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 4 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 4 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using SmemLayoutAtomA = decltype(composition(
        Swizzle<2, 4, 3>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutAtomB = decltype(composition(
        Swizzle<2, 4, 3>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))));

    using s2r_copy_op_a = SM75_U32x4_LDSM_N;
    using s2r_copy_traits_a = Copy_Traits<s2r_copy_op_a>;
    using s2r_copy_atom_a = Copy_Atom<s2r_copy_traits_a, uint8_t>;

    using s2r_copy_op_b = SM75_U32x4_LDSM_N;
    using s2r_copy_traits_b = Copy_Traits<s2r_copy_op_b>;
    using s2r_copy_atom_b = Copy_Atom<s2r_copy_traits_b, uint8_t>;

    using SmemCopyAtomA = s2r_copy_atom_a;
    using SmemCopyAtomB = s2r_copy_atom_b;

    using GmemTiledCopy = decltype(  
        make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint128_t>, ElementA>{},  
                        Layout<Shape <_16,_8>,
                            Stride< _8,_1>>{},
                        Layout<Shape < _1,_16>>{}));
 
    using GmemTiledCopyA = GmemTiledCopy;
    using GmemTiledCopyB = GmemTiledCopy;

    using CollectiveMainloop = collective::CollectiveMma<  
        DispatchPolicy, TileShape,  
        ElementA, TagToStrideA_t<LayoutA>,  
        ElementB, TagToStrideB_t<LayoutB>,  
        MMA,  
        GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  
        GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity  
    >;  
  
    using SmemLayoutCAtom = decltype(composition(
        Swizzle<3, 2, 3>{},
        make_layout(make_shape(Int<8>{}, Int<kTileN>{}),
                    make_stride(Int<kTileN>{}, Int<1>{}))));
    using SmemLayoutC = decltype(
        tile_to_shape(SmemLayoutCAtom{},
                        make_shape(Int<kTileM>{}, Int<kTileN>{})));

    using CollectiveEpilogue = epilogue::collective::Epilogue<
        TagToStrideC_t<LayoutC>, TagToStrideC_t<LayoutC>,
        epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
        SmemLayoutC,
        Copy_Atom<UniversalCopy<uint32_t>, int32_t>,                           // R2S with tiled_mma layout
        decltype(make_tiled_copy(Copy_Atom<UniversalCopy<int32_t>,int32_t>{}, // S2R
                                Layout<Shape <_16,_8>,
                                        Stride< _8,_1>>{},
                                Layout<Shape<_1,_4>>{})),
        Copy_Atom<UniversalCopy<uint128_t>,int32_t>                           // R2G with S2R_dst layout
        >;
  
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<  
      Shape<int,int,int>,  
      CollectiveMainloop,  
      CollectiveEpilogue  
    >;  
  
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // print(GmemTiledCopyA{});
    // printf("\n"); 
    // print(SmemLayoutA{});
    // printf("\n"); 
    // print(make_tiled_copy_A(s2r_copy_atom{}, MMA{}));
    // print_latex(typename s2r_copy_traits::DstLayout{});
    // print_latex(SmemLayoutA{});
    // print(MMA{});
    // print_latex(SmemLayoutAtom{});

    std::srand(42);

    // 初始化参数  
    int32_t alpha = 1, beta = 0;  
  
    // 创建主机张量  
    cutlass::HostTensor<ElementA, LayoutA> tensor_A({m, k});  
    cutlass::HostTensor<ElementB, LayoutB> tensor_B({k, n});  // 注意B是ColumnMajor  
    cutlass::HostTensor<ElementC, LayoutC> tensor_C({m, n});  
    cutlass::HostTensor<ElementC, LayoutC> tensor_D({m, n});  
    cutlass::HostTensor<ElementC, LayoutC> reference_D({m, n});  
  
    // 填充输入数据  
    cutlass::reference::host::TensorFillRandomUniform(  
        tensor_A.host_view(), 1, ElementA(4), ElementA(-4), 0);  
    cutlass::reference::host::TensorFillRandomUniform(  
        tensor_B.host_view(), 1, ElementB(4), ElementB(-4), 0);  
    cutlass::reference::host::TensorFillRandomUniform(  
        tensor_C.host_view(), 1, ElementC(4), ElementC(-4), 0);  
  
    // 复制C到D和reference_D  
    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());  
    cutlass::reference::host::TensorCopy(tensor_D.host_view(), tensor_C.host_view());  
  
    // 同步到设备  
    tensor_A.sync_device();  
    tensor_B.sync_device();  
    tensor_C.sync_device();  
    tensor_D.sync_device();

    using StrideA = Stride<int64_t, Int<1>, int64_t>;
    using StrideB = Stride<int64_t, Int<1>, int64_t>;
    using StrideC = Stride<int64_t, Int<1>, int64_t>;

    StrideA a_stride{tensor_A.layout().stride(0), Int<1>{}, 0};
    StrideB b_stride{tensor_B.layout().stride(0), Int<1>{}, 0};
    StrideC c_stride{tensor_C.layout().stride(0), Int<1>{}, Int<0>{}};
  
    // 设置GEMM参数  
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,  // 替换GemmCoord为GemmUniversalMode
        {m, n, k},  // 问题尺寸
        {tensor_A.device_data(), a_stride, tensor_B.device_data(), b_stride},
        {{static_cast<ElementAccumulator>(alpha), static_cast<ElementAccumulator>(beta)}, tensor_D.device_data(), c_stride, tensor_D.device_data(), c_stride}, 
    };
  
    // 初始化GEMM操作  
    Gemm gemm_op;  
    cutlass::Status status = gemm_op.initialize(arguments);  
      
    if (status != cutlass::Status::kSuccess) {  
        std::cerr << "Failed to initialize GEMM: " << cutlass::cutlassGetStatusString(status) << std::endl;  
        return -1;  
    }  
  
    // 执行GEMM  
    status = gemm_op();
      
    if (status != cutlass::Status::kSuccess) {  
        std::cerr << "Failed to run GEMM: " << cutlass::cutlassGetStatusString(status) << std::endl;  
        return -1;  
    }  
  
    // 计算参考结果  
    cutlass::reference::host::Gemm<  
        ElementA, LayoutA,  
        ElementB, LayoutB,  
        ElementC, LayoutC,   
        int32_t, int32_t>  
        reference_gemm;  
  
    reference_gemm(  
        {m, n, k},  
        alpha,  
        tensor_A.host_ref(),  
        tensor_B.host_ref(),  
        beta,  
        reference_D.host_ref(),  
        int32_t(0)  
    );  
  
    // 同步结果  
    tensor_D.sync_host();  
  
    // 验证结果  
    bool passed = cutlass::reference::host::TensorEquals(  
        reference_D.host_view(),  
        tensor_D.host_view()  
    );  
  
    if (passed) { 
        std::cout << "GEMM verification PASSED!" << std::endl; 
    } else { 
        std::cout << "GEMM verification FAILED!" << std::endl; 
        // return -1; 
    }

    std::cout << std::endl;
    // return 0;

    // 打印tensor_D的内容
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            std::cout << tensor_D.host_view().at({i, j}) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    // 打印reference_D的内容
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            std::cout << reference_D.host_view().at({i, j}) << " ";
        }
        std::cout << std::endl;
    }

    float gflops = (2.0*m*n*k) * 1e-9;
    float gBs = (m*k + m*n*4 + n*k) * 1e-9;
    const int timing_warmup_iterations = 10;
    const int timing_iterations = 100;
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < timing_warmup_iterations; ++i) {
        status = gemm_op();  
    }
    timer.seconds();

    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        status = gemm_op();  
    }
    float cute_time = timer.seconds() / timing_iterations;
    printf("CUTE_GEMM:     [%6.1f]GFlop/s [%6.1f]GB/s  (%6.4f)ms\n", gflops / cute_time, gBs / cute_time, cute_time*1000);

    CublasLtGemm<int8_t, int32_t> cublas_gemm;
    cublas_gemm.init(tensor_C.device_data(), tensor_A.device_data(), tensor_B.device_data(), m, n, k);

    timer.start();
    for (int i = 0; i < timing_warmup_iterations; ++i) {
        cublas_gemm.run();
    }
    timer.seconds();

    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        cublas_gemm.run();
    }
    float cublaslt_time = timer.seconds() / timing_iterations;
    printf("CUBLAS_GEMM:   [%6.1f]GFlop/s [%6.1f]GB/s  (%6.4f)ms\n", gflops / cublaslt_time, gBs / cublaslt_time, cublaslt_time*1000);

    return 0;  
}
