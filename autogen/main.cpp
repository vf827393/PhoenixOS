/*
 * Copyright 2024 The PhoenixOS Authors. All rights reserved.
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

#include "autogen_common.h"

int main(int argc, char** argv) {
    int opt;
    const char *op_string = "d:s:g:";
    pos_retval_t retval = POS_SUCCESS;
    std::string token;
    std::stringstream ss;

    POSAutogener autogener;

    while((opt = getopt(argc, argv, op_string)) != -1){
        switch (opt){
        case 'd':
            // path to the vendor header files
            autogener.all_vendor_header_directories_str = std::string(optarg);
            ss.str(autogener.all_vendor_header_directories_str);
            while (std::getline(ss, token, ',')) {
                autogener.vendor_header_directories.push_back(token);
            }
            break;
        case 's':
            // path to the support files
            autogener.support_directory = std::string(optarg);
            break;
        case 'g':
            // path to generate the parser and worker logic
            autogener.gen_directory = std::string(optarg);
            break;
        default:
            POS_ERROR("unknown command line parameter: %s", op_string);
        }
    }

    // check whether all necessary parameters are provided
    if(unlikely(autogener.vendor_header_directories.size() == 0)){
        POS_ERROR("no vendor header directories provided with -d");
    }
    if(unlikely(autogener.support_directory.size() == 0)){
        POS_ERROR("no support_directory provided with -s");
    }
    if(unlikely(autogener.gen_directory.size() == 0)){
        POS_ERROR("no gen_directory provided with -g");
    }

    if(unlikely(
        POS_SUCCESS != (retval = autogener.collect_pos_support_yamls())
    )){
        POS_WARN("failed to collect PhOS support metadata");
        goto exit;
    }

    if(unlikely(
        POS_SUCCESS != (retval = autogener.collect_vendor_header_files())
    )){
        POS_WARN("failed to parse vendor headers: path(%s)", autogener.all_vendor_header_directories_str.c_str());
        goto exit;
    }

    if(unlikely(
        POS_SUCCESS != (retval = autogener.generate_pos_src())
    )){
        POS_WARN("failed to auto-generate source code");
        goto exit;
    }

exit:
    if(unlikely(retval != POS_SUCCESS))
        return -1;
    else
        return 0;
}
