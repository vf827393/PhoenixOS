#include <iostream>
#include "pos/include/common.h"
#include "pos/include/client.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "pos/cuda_impl/worker.h"


namespace wk_functions {
namespace cuda_get_error_string {
    POS_WK_FUNC_LAUNCH() {
        pos_retval_t retval = POS_SUCCESS;

        POS_CHECK_POINTER(wqe);
        POS_CHECK_POINTER(ws);

        #if POS_CONF_RUNTIME_EnableDebugCheck
            POS_ASSERT(wqe->input_handle_views.size() == 0);
            POS_ASSERT(wqe->output_handle_views.size() == 0);
            POS_ASSERT(wqe->inout_handle_views.size() == 0);
        #endif

        // TODO: we need to fix this issue, no return value
        cudaGetErrorString(
            /* error */ (cudaError_t)(pos_api_param_value(wqe, 0, cudaError_t))
        );

        if(unlikely(cudaSuccess != wqe->api_cxt->return_code)){
           POSWorker::__restore(ws, wqe);
        } else {
           POSWorker::__done(ws, wqe);
        }

     exit:

        return retval;
    }

} // namespace cuda_get_error_string

} // namespace wk_functions
