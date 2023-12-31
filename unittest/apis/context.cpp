#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pos/common.h"
#include "pos/log.h"
#include "pos/unittest/apis/base.h"
#include "pos/unittest/unittest.h"
#include "pos/unittest/include/utils.h"

pos_retval_t test_cuda_get_last_error(test_cxt* cxt){
    pos_retval_t retval = POS_SUCCESS;
    cudaError cuda_result;
    uint64_t s_tick, e_tick;

    s_tick = pos_utils_get_tsc();
    cuda_result = cudaGetLastError();
    e_tick = pos_utils_get_tsc();
    
    cxt->duration_ticks = e_tick - s_tick;

    if(unlikely(cuda_result != cudaSuccess)){
        retval = POS_FAILED;
        goto exit;
    }

exit:
    return retval;
}

pos_retval_t test_cuda_get_error_string(test_cxt* cxt){
    pos_retval_t retval = POS_SUCCESS;
    const char* str = nullptr;
    uint64_t s_tick, e_tick;

    s_tick = pos_utils_get_tsc();
    str = cudaGetErrorString(cudaErrorDeviceUninitialized);
    e_tick = pos_utils_get_tsc();
    
    cxt->duration_ticks = e_tick - s_tick;

    if(unlikely(str == nullptr)){
        retval = POS_FAILED;
        goto exit;
    }
    
    if(unlikely(strcmp("invalid device context", str))){
        retval = POS_FAILED;
        goto exit;
    }

exit:
    return retval;
}
