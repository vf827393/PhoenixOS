#pragma once

#include <iostream>
#include <vector>
#include <stdint.h>

#include <iostream>

#include <stdint.h>

// TSC frequency
#define POS_TSC_FREQ 2200000000 // Hz

#define POS_TSC_RANGE_TO_SEC(e_tick, s_tick) \
    (double)(e_tick - s_tick) / (double) POS_TSC_FREQ

#define POS_TSC_RANGE_TO_MSEC(e_tick, s_tick) \
    (double)(e_tick - s_tick) / (double) POS_TSC_FREQ * (double)1000.0f

#define POS_TSC_RANGE_TO_USEC(e_tick, s_tick) \
    (double)(e_tick - s_tick) / (double) POS_TSC_FREQ * (double)1000000.0f

#define POS_TSC_TO_USEC(tick) \
    (double)(tick) / (double) POS_TSC_FREQ * (double)1000000.0f

uint64_t pos_utils_get_tsc();
