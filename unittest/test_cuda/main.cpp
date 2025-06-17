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

#include "test_cuda_common.h"

/*
    this listener ensures that the process exits immediately once any test case fails 
*/

class AbortOnFailureListener : public testing::EmptyTestEventListener {

    void OnTestStart(const testing::TestInfo& test_info) override {
        printf("*** Test %s.%s starting.\n",
        test_info.test_suite_name(), test_info.name());
    }
  
    void OnTestPartResult(const testing::TestPartResult& test_part_result) override {
        printf("%s in %s:%d\n%s\n",
            test_part_result.failed() ? "*** Failure" : "Success",
            test_part_result.file_name(),
            test_part_result.line_number(),
            test_part_result.summary());

        if (test_part_result.failed()) {
        std::cerr << "\n💥 Test failure detected!\n";
        std::cerr << "  Location: " << test_part_result.file_name() << ":" << test_part_result.line_number() << "\n";
        std::cerr << "  Message : " << test_part_result.summary() << "\n";
        std::abort();  
     }
    }

    void OnTestEnd(const testing::TestInfo& test_info) override {
            printf("*** Test %s.%s ending.\n",
            test_info.test_suite_name(), test_info.name());
    }
};

int main(int argc, char *argv[]){
    ::testing::InitGoogleTest(&argc, argv);

    // delete default listener
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());

    // add my own listener
    listeners.Append(new AbortOnFailureListener);
    printf("run test");
    return RUN_ALL_TESTS();
}
