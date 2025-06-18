#include "test_cuda/test_cuda_common.h"


TEST_F(PhOSCudaTest, cuModuleGetFunction) {
    CUresult cuda_retval;
    CUmodule module;
    CUmodule *module_ptr = &module;
    CUfunction function;
    CUfunction *function_ptr = &function;
    std::ifstream in;
    std::stringstream buffer;
    std::string function_name;
    const char* function_name_ptr;

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path current_abs_path = std::filesystem::absolute(current_path);
    std::filesystem::path current_dir_abs_path = current_abs_path.parent_path();
    std::filesystem::path current_dir_dir_abs_path = current_dir_abs_path.parent_path();

    #if CUDA_VERSION >= 9000 && CUDA_VERSION < 11040
        std::filesystem::path cubin_asb_path = std::filesystem::canonical(
            current_dir_dir_abs_path / "assets" / "sm70_72_75_80_86.fatbin"
        );
    #else
        POS_WARN("no test file for current cuda architecture: cuda_version(%d)", CUDA_VERSION);
        goto exit;
    #endif

    in.open(cubin_asb_path, std::ios::binary);
    EXPECT_EQ(true, in.is_open());
    buffer << in.rdbuf();

    // load module first
    cuda_retval = (CUresult)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cuModuleLoadData, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            { .value = &module_ptr, .size = sizeof(CUmodule*) },
            { .value = buffer.str().data(), .size = buffer.str().size() }
        }
    );
    EXPECT_EQ(CUDA_SUCCESS, cuda_retval);

    function_name = "_Z15squareMatrixMulPKiS0_Pii";
    function_name_ptr = function_name.data();

    // get function
    cuda_retval = (CUresult)this->_ws->pos_process( 
        /* api_id */ PosApiIndex_cuModuleGetFunction, 
        /* uuid */ this->_clnt->id,
        /* param_desps */ { 
            { .value = &function_ptr, .size = sizeof(CUfunction*) },
            { .value = &module, .size = sizeof(CUmodule) },
            { .value = &function_name_ptr, .size = sizeof(const char*) }
        }
    );
    EXPECT_EQ(CUDA_SUCCESS, cuda_retval);

exit:
    if(in.is_open()){
        in.close();
    }
}
