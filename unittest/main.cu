#include <iostream>
#include <vector>

#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pos/common.h"
#include "pos/log.h"
#include "pos/unittest/unittest.h"

int main(){
    std::map<uint64_t, pos_retval_t> result_map;

    POSUnitTest ut;
    ut.run_all();

    return 0;
}
