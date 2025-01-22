/*
 * Copyright 2025 The PhoenixOS Authors. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <filesystem>


#include "pos/include/common.h"
#include "pos/include/log.h"


class POSUtilSystem {
 public:
    POSUtilSystem(){}
    ~POSUtilSystem(){}

    /* ======================== Memory ======================== */
 public:
    static pos_retval_t get_memory_info(uint64_t& total_bytes, uint64_t& avail_bytes){
        pos_retval_t retval = POS_SUCCESS;
        std::ifstream memInfo("/proc/meminfo");
        std::string line;

        if (!std::filesystem::exists("/proc/meminfo")) {
            POS_WARN("failed to get memory info, /proc/meminfo not exists");
            retval = POS_FAILED_NOT_EXIST;
            goto exit;
        }

        while (std::getline(memInfo, line)) {
            if (line.find("MemTotal:") == 0) {
                total_bytes = std::stoll(line.substr(line.find_first_of("0123456789")));
            } else if (line.find("MemAvailable:") == 0) {
                avail_bytes = std::stoll(line.substr(line.find_first_of("0123456789")));
                break; // No need to read more lines
            }
        }

        total_bytes *= 1024;
        avail_bytes *= 1024;

    exit:
        return retval;
    }

    /*!
     *  \brief  format a byte number into a string with unit
     *  \param  bytes   byte number
     *  \return string with unit
     */
    static std::string format_byte_number(uint64_t bytes){
        const std::string suffixes[] = {"B", "K", "M", "G"};
        int index = 0;
        double bytes_d = static_cast<double>(bytes);

        while (bytes_d >= 1024 && index < 3) {
            bytes_d /= 1024;
            index++;
        }

        bytes_d = std::ceil(bytes_d);
        return std::to_string(static_cast<int>(bytes_d)) + suffixes[index];
    }    
    /* ======================== Memory ======================== */
};
