#include <iostream>
#include <vector>

#include <stdint.h>

#include "pos/unittest/include/utils.h"

uint64_t pos_utils_get_tsc(){
    uint64_t a, d;
    __asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
    return (d << 32) | a;
}
