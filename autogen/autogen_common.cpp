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

#include <iostream>
#include <algorithm>
#include <string>

#include "autogen_common.h"


pos_handle_source_typeid_t get_handle_source_by_name(std::string& handle_source){
    if(handle_source == std::string("from_param")){
        return kPOS_HandleSource_FromParam;
    } else if(handle_source == std::string("to_param")){
        return kPOS_HandleSource_ToParam;
    } else if(handle_source == std::string("from_last_used")){
        return kPOS_HandleSource_FromLastUsed;
    } else if(handle_source == std::string("from_default")){
        return kPOS_HandleSource_FromDefault;
    } else {
        POS_ERROR_DETAIL(
            "invalid handle source detected: given_handle_source(%s)", handle_source.c_str()
        );
    }
}


pos_edge_side_effect_typeid_t get_side_effect_by_name(std::string& side_effect){
    if(side_effect == std::string("set_as_last_used")){
        return kPOS_EdgeSideEffect_SetAsLastUsed;
    } else {
        POS_ERROR_DETAIL(
            "invalid side effect detected: given_side_effect(%s)", side_effect.c_str()
        );
    }
}


pos_retval_t POSAutogener::collect_pos_support_yamls(){
    pos_retval_t retval = POS_SUCCESS;
    pos_support_header_file_meta_t *header_file_meta;
    std::string library_name;

    POS_ASSERT(this->support_directory.size() > 0);

    if(unlikely(
            !std::filesystem::exists(this->support_directory)
        ||  !std::filesystem::is_directory(this->support_directory)
    )){
        POS_WARN_C(
            "failed to do autogen, invalid support files path provided: path(%s)",
            this->support_directory.c_str()
        );
        retval = POS_FAILED_INVALID_INPUT;
        goto exit;
    }

    for (const auto& entry : std::filesystem::recursive_directory_iterator(this->support_directory)){
        if(entry.is_regular_file() && entry.path().extension() == ".yaml"){
            POS_LOG_C("parsing support file %s...", entry.path().c_str())

            library_name = entry.path().parent_path().filename().string();

            retval = this->__try_get_header_file_meta(entry.path(), &header_file_meta);
            if(retval == POS_SUCCESS){
                POS_CHECK_POINTER(header_file_meta);
            } else if(retval == POS_FAILED_NOT_EXIST) {
                POS_CHECK_POINTER(header_file_meta = new pos_support_header_file_meta_t);
            } else {
                POS_ERROR_C("failed to parse file %s", entry.path().c_str())
            }
            
            if(unlikely(POS_SUCCESS != (
                retval = this->__collect_pos_support_yaml(
                    entry.path(),
                    header_file_meta,
                    retval == POS_FAILED_NOT_EXIST ? true : false,
                    library_name
                )
            ))){
                POS_ERROR_C("failed to parse file %s", entry.path().c_str())
            }

            /// !   \note   the file_name of header_file_meta should be updated in __collect_pos_support_yaml
            if(unlikely(header_file_meta->file_name.size() == 0)){
                POS_WARN_C("no name header file provided in yaml file: path(%s)", entry.path().c_str());
                retval = POS_FAILED_INVALID_INPUT;
                goto exit;
            }
            this->_supported_header_file_meta_map.insert({ header_file_meta->file_name , header_file_meta });
        }
    }

exit:
    return retval;
}


pos_retval_t POSAutogener::collect_vendor_header_files(){
    pos_retval_t retval = POS_SUCCESS;
    uint64_t i;
    pos_vendor_header_file_meta_t *vendor_header_file_meta;
    pos_support_header_file_meta_t *supported_header_file_meta;
    typename std::map<std::string, pos_support_header_file_meta_t*>::iterator header_map_iter;
    typename std::map<std::string, pos_support_api_meta_t*>::iterator api_map_iter;
    pos_support_api_meta_t *support_api_meta;

    POS_ASSERT(this->vendor_header_directories.size() > 0);

    auto __collect_vendor_header_file = [this](std::string& header_file_path) -> pos_retval_t {
        pos_retval_t retval = POS_SUCCESS;
        pos_support_header_file_meta_t *supported_header_file_meta;
        pos_vendor_header_file_meta_t *vendor_header_file_meta;
        std::string relative_path;

        if(unlikely(
                !std::filesystem::exists(header_file_path)
            ||  !std::filesystem::is_directory(header_file_path)
        )){
            POS_WARN_C(
                "failed to do autogen, invalid vender headers path provided: path(%s)",
                header_file_path.c_str()
            );
            retval = POS_FAILED_INVALID_INPUT;
            goto exit;
        }

        for(const auto& entry : std::filesystem::recursive_directory_iterator(header_file_path)){
            if(entry.is_regular_file() && (entry.path().extension() == ".h" || entry.path().extension() == ".hpp")){
                POS_LOG_C("parsing vendor header file %s...", entry.path().c_str());
            } else {
                continue;
            }

            // calculate the diff of paths, e.g.,
            // header_file_path == /usr/local/cuda-11.3/include
            // entry.path() == /usr/local/cuda-11.3/include/crt/host_runtime.h
            // the diff should be crt/host_runtime.h
            relative_path = std::filesystem::path(entry.path()).lexically_relative(header_file_path);

            if(this->_supported_header_file_meta_map.count(relative_path) == 0){
                POS_BACK_LINE;
                POS_OLD_LOG_C("parsing vendor header file %s [skipped]", entry.path().c_str());
                continue;
            }
            POS_CHECK_POINTER(supported_header_file_meta = this->_supported_header_file_meta_map[relative_path]);

            POS_CHECK_POINTER(vendor_header_file_meta = new pos_vendor_header_file_meta_t);
            vendor_header_file_meta->file_name = relative_path;
            this->_vendor_header_file_meta_map.insert(
                { vendor_header_file_meta->file_name, vendor_header_file_meta }
            );

            if(unlikely(POS_SUCCESS != (
                retval = this->__collect_vendor_header_file(
                    entry.path(),
                    vendor_header_file_meta,
                    supported_header_file_meta
                )
            ))){
                POS_ERROR_C("failed to parse file %s", entry.path().c_str())
            }
        }

    exit:
        return retval;
    };


    auto __parse_api_prototype = [this](
        std::string& prototype,
        pos_vendor_header_file_meta_t *vendor_header_file_meta
    ) -> pos_retval_t {
        pos_retval_t retval = POS_SUCCESS;
        CXIndex index;
        CXErrorCode cx_retval;
        CXTranslationUnit unit;
        CXCursor cursor;
        CXUnsavedFile unsaved_file;

        POS_CHECK_POINTER(vendor_header_file_meta);

        unsaved_file.Filename = "temp.cpp";
        unsaved_file.Contents = prototype.c_str();
        unsaved_file.Length = static_cast<unsigned long>(prototype.length());

        index = clang_createIndex(0, 0);

        cx_retval = clang_parseTranslationUnit2(
            /* CIdx */ index,
            /* source_filename */ "temp.cpp",
            /* command_line_args */ nullptr,
            /* nb_command_line_args */ 0,
            /* unsaved_files */ &unsaved_file,
            /* nb_unsaved_file */ 1,
            /* options */ CXTranslationUnit_None,
            /* out_TU */ &unit
        );
        if(unlikely(cx_retval != CXError_Success)){
            POS_WARN_DETAIL(
                "failed to parse the function prototype: prototype(%s)",
                prototype.c_str()
            );
            retval = POS_FAILED;
            goto __exit;
        }
        if(unlikely(unit == nullptr)){
            POS_ERROR_DETAIL("failed to create clang translation unit");
        }

        cursor = clang_getTranslationUnitCursor(unit);
        clang_visitChildren(
            /* parent */ cursor,
            /* visitor */
            [](CXCursor cursor, CXCursor parent, CXClientData client_data) -> CXChildVisitResult {
                int i, num_args;
                std::string func_name_cppstr;
                CXString func_name;
                CXCursor arg_cursor;
                pos_vendor_header_file_meta_t *vendor_header_file_meta = nullptr;
                pos_vendor_api_meta_t *api_meta = nullptr;
                pos_vendor_param_meta_t *param_meta = nullptr;

                if (clang_getCursorKind(cursor) == CXCursor_FunctionDecl) {
                    vendor_header_file_meta = reinterpret_cast<pos_vendor_header_file_meta_t*>(client_data);
                    POS_CHECK_POINTER(vendor_header_file_meta);

                    func_name_cppstr = std::string(clang_getCString(clang_getCursorSpelling(cursor)));

                    POS_CHECK_POINTER(api_meta = new pos_vendor_api_meta_t);
                    vendor_header_file_meta->api_map.insert({ func_name_cppstr, api_meta });
                    api_meta->name = std::string(clang_getCString(clang_getCursorSpelling(cursor)));
                    api_meta->return_type = std::string(clang_getCString(clang_getTypeSpelling(clang_getCursorResultType(cursor))));
                    // returnType = clang_getTypeSpelling(clang_getCursorResultType(cursor));

                    num_args = clang_Cursor_getNumArguments(cursor);
                    for(i=0; i<num_args; i++){
                        POS_CHECK_POINTER(param_meta = new pos_vendor_param_meta_t);
                        api_meta->params.push_back(param_meta);
                        arg_cursor = clang_Cursor_getArgument(cursor, i);
                        param_meta->name = std::string(clang_getCString(clang_getCursorSpelling(arg_cursor)));
                        param_meta->type = std::string(clang_getCString(clang_getTypeSpelling(clang_getCursorType(arg_cursor))));
                        param_meta->is_pointer = clang_getCursorType(arg_cursor).kind == CXType_Pointer;
                    }
                }

            cursor_traverse_exit:
                return CXChildVisit_Recurse;
            },
            /* client_data */ vendor_header_file_meta
        );

        clang_disposeTranslationUnit(unit);

    __exit:
        return retval;
    };


    // collect vendor header files
    for(i=0; i<this->vendor_header_directories.size(); i++){
        retval = __collect_vendor_header_file(this->vendor_header_directories[i]);
        if(retval != POS_SUCCESS){
            POS_WARN_C("failed to parse vendor header file %s", this->vendor_header_directories[i].c_str());
            goto exit;
        }
    }

    // check
    for(header_map_iter = this->_supported_header_file_meta_map.begin();
        header_map_iter != this->_supported_header_file_meta_map.end();
        header_map_iter++
    ){
        // [1] whether all header files that registered as supported were founded
        const std::string &supported_file_name = header_map_iter->first;

        if(unlikely(this->_vendor_header_file_meta_map.count(supported_file_name) == 0)){
            POS_WARN_C(
                "PhOS registered to support header file %s, but no vendor header file was found",
                supported_file_name.c_str()
            );
            retval = POS_FAILED_NOT_EXIST;
            goto exit;
        }

        vendor_header_file_meta = this->_vendor_header_file_meta_map[supported_file_name];
        POS_CHECK_POINTER(vendor_header_file_meta);
        supported_header_file_meta = header_map_iter->second;
        POS_CHECK_POINTER(supported_header_file_meta);

        // [2] whether all apis that registered as supported were founded
        for(api_map_iter = supported_header_file_meta->api_map.begin();
            api_map_iter != supported_header_file_meta->api_map.end();
            api_map_iter++
        ){
            POS_CHECK_POINTER(support_api_meta = api_map_iter->second);
            if(unlikely(
                vendor_header_file_meta->api_map.count(support_api_meta->parent_name) == 0
            )){
                if(support_api_meta->prototype.size() == 0){
                    POS_WARN_C(
                        "PhOS registered to support API %s in file %s, but neither vendor API or prototype in yaml was found",
                        support_api_meta->parent_name.c_str(), supported_file_name.c_str()
                    );
                    retval = POS_FAILED_NOT_EXIST;
                    goto exit;
                } else {
                    // POS_ASSERT(support_api_meta->parser_type == "customized" and support_api_meta->worker_type == "customized");

                    // we append the clang parsing of the API here
                    retval = __parse_api_prototype(support_api_meta->prototype, vendor_header_file_meta);
                    if(unlikely(retval != POS_SUCCESS)){
                        POS_WARN_C(
                            "PhOS registered to support API %s via given prototype \"%s\", yet failed to parse the prototype",
                            support_api_meta->parent_name.c_str(),
                            support_api_meta->prototype.c_str()
                        );
                        goto exit;
                    }
                }
            }
        }
    }

exit:
    return retval;
}

pos_retval_t POSAutogener::generate_pos_src(){
    pos_retval_t retval = POS_SUCCESS;
    pos_vendor_header_file_meta_t *vendor_header_file_meta = nullptr;
    pos_support_header_file_meta_t *supported_header_file_meta = nullptr;
    pos_vendor_api_meta_t *vendor_api_meta = nullptr;
    pos_support_api_meta_t *support_api_meta = nullptr;

    typename std::map<std::string, pos_support_header_file_meta_t*>::iterator header_map_iter;
    typename std::map<std::string, pos_support_api_meta_t*>::iterator api_map_iter;

    std::vector<pos_vendor_header_file_meta_t*> vendor_header_file_meta_list;
    std::vector<pos_support_api_meta_t*> support_api_meta_list;

    // recreate generate folders
    this->parser_directory = this->gen_directory + std::string("/parser");
    this->worker_directory = this->gen_directory + std::string("/worker");
    try {
        if (std::filesystem::exists(this->gen_directory)) { std::filesystem::remove_all(this->gen_directory); }
        std::filesystem::create_directories(this->gen_directory);
        std::filesystem::create_directories(this->parser_directory);
        std::filesystem::create_directories(this->worker_directory);
    } catch (const std::filesystem::filesystem_error& e) {
        POS_WARN_C(
            "failed to create new directory for the generated codes: parser_directory(%s), worker_directory(%s)",
            this->parser_directory.c_str(), this->worker_directory.c_str()
        );
        retval = POS_FAILED;
        goto exit;
    } catch (const std::exception& e) {
        POS_WARN_C(
            "failed to create new directory for the generated codes: parser_directory(%s), worker_directory(%s)",
            this->parser_directory.c_str(), this->worker_directory.c_str()
        );
        retval = POS_FAILED;
        goto exit;
    }

    // iterate through all APIs
    for(header_map_iter = this->_supported_header_file_meta_map.begin();
        header_map_iter != this->_supported_header_file_meta_map.end();
        header_map_iter++
    ){
        const std::string &supported_file_name = header_map_iter->first;

        vendor_header_file_meta = this->_vendor_header_file_meta_map[supported_file_name];
        POS_CHECK_POINTER(vendor_header_file_meta);
        supported_header_file_meta = header_map_iter->second;
        POS_CHECK_POINTER(supported_header_file_meta);

        vendor_header_file_meta_list.push_back(vendor_header_file_meta);

        for(api_map_iter = supported_header_file_meta->api_map.begin();
            api_map_iter != supported_header_file_meta->api_map.end();
            api_map_iter++
        ){
            const std::string &api_name = api_map_iter->first;
            support_api_meta = api_map_iter->second;
            POS_CHECK_POINTER(support_api_meta);

            support_api_meta_list.push_back(support_api_meta);

            vendor_api_meta = vendor_header_file_meta->api_map[support_api_meta->parent_name];
            POS_CHECK_POINTER(vendor_api_meta);

            // generate parser logic
            POS_LOG_C("generating parser logic for API %s...", api_name.c_str());
            if(unlikely(POS_SUCCESS != (
                retval = this->__generate_api_parser(vendor_api_meta, support_api_meta)
            ))){
                POS_ERROR_C("generating parser logic for API %s..., failed", api_name.c_str());
            }
            POS_BACK_LINE;
            POS_LOG_C("generating parser logic for API %s: [done]", api_name.c_str());

            // generate worker logic
            POS_LOG_C("generating worker logic for API %s...", api_name.c_str());
            if(unlikely(POS_SUCCESS != (
                retval = this->__generate_api_worker(
                    vendor_header_file_meta,
                    supported_header_file_meta,
                    vendor_api_meta,
                    support_api_meta
                )
            ))){
                POS_ERROR_C("generating worker logic for API %s..., failed", api_name.c_str());
            }
            POS_BACK_LINE;
            POS_LOG_C("generating worker logic for API %s: [done]", api_name.c_str());
        }
    }

    // generate index macro file
    if(unlikely(POS_SUCCESS != (
        retval = this->__generate_auxiliary_files(vendor_header_file_meta_list, support_api_meta_list)
    ))){
        POS_ERROR_C("generating index macro file failed");
    }

exit:
    return retval;
}


pos_retval_t POSAutogener::__generate_api_parser(
    pos_vendor_api_meta_t* vendor_api_meta,
    pos_support_api_meta_t* support_api_meta
){
    pos_retval_t retval = POS_SUCCESS;
    uint64_t i;
    POSCodeGen_CppSourceFile *parser_file;
    POSCodeGen_CppBlock *ps_function_namespace, *api_namespace, *parser_function;
    std::string api_snake_name;
    std::string parser_file_directory;

    POS_CHECK_POINTER(vendor_api_meta);
    POS_CHECK_POINTER(support_api_meta);

    // for those APIs to be customized logic, we just omit
    if(support_api_meta->parser_type == std::string("customized")){
        goto exit;
    }

    api_snake_name = posautogen_utils_camel2snake(support_api_meta->name);

    parser_file_directory = this->parser_directory + std::string("/") + support_api_meta->library_name;
    if(!std::filesystem::exists(parser_file_directory)){
        try {
            std::filesystem::create_directory(parser_file_directory);
        } catch (const std::exception& e) {
            POS_WARN_C(
                "failed to create new library directory for the generated codes: parser_file_directory(%s)",
                parser_file_directory.c_str()
            );
            retval = POS_FAILED;
            goto exit;
        }
    }

    // create parser file
    parser_file = new POSCodeGen_CppSourceFile(
        parser_file_directory
        + std::string("/")
        + support_api_meta->name
        + std::string(".cpp")
    );
    POS_CHECK_POINTER(parser_file);
    
    // add basic headers to the parser file
    parser_file->add_preprocess("#include <iostream>");
    parser_file->add_preprocess("#include \"pos/include/common.h\"");
    for(i=0; i<support_api_meta->dependent_headers.size(); i++){
        parser_file->add_preprocess(std::format("#include <{}>", support_api_meta->dependent_headers[i]));
    }

    // create ps_function namespace
    ps_function_namespace = new POSCodeGen_CppBlock(
        /* field name */ "namespace ps_functions",
        /* need_braces */ true,
        /* foot_comment */ "namespace ps_functions"
    );
    POS_CHECK_POINTER(ps_function_namespace);
    parser_file->add_block(ps_function_namespace);

    // create api namespace
    retval = ps_function_namespace->allocate_block(
        /* field name */ std::string("namespace ") + api_snake_name,
        /* new_block */ &api_namespace,
        /* need_braces */ true,
        /* foot_comment */ std::string("namespace ") + api_snake_name,
        /* need_ended_semicolon */ false,
        /* level_offset */ 0
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to allocate cpp block for api namespace while generating parser function: "
            "api_name(%s)",
            api_snake_name
        );
    }
    POS_CHECK_POINTER(api_namespace);

    // create function POS_RT_FUNC_PARSER
    retval = api_namespace->allocate_block(
        /* field name */ std::string("POS_RT_FUNC_PARSER()"),
        /* new_block */ &parser_function,
        /* need_braces */ true,
        /* foot_comment */ "",
        /* need_ended_semicolon */ false,
        /* level_offset */ 1
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to allocate cpp block for POS_RT_FUNC_PARSER while generating parser function: "
            "api_name(%s)",
            api_snake_name
        );
    }
    POS_CHECK_POINTER(parser_function);

    if(unlikely(POS_SUCCESS != (
        retval = this->__insert_code_parser_for_target(
            vendor_api_meta,
            support_api_meta,
            parser_file,
            ps_function_namespace,
            api_namespace,
            parser_function
        )
    ))){
        POS_WARN_C("failed to generate parser when inserted target-specific code");
        goto exit;
    }

    parser_file->archive();

exit:
    return retval;
}


pos_retval_t POSAutogener::__generate_api_worker(
    pos_vendor_header_file_meta_t* vender_header_file_meta,
    pos_support_header_file_meta_t* support_header_file_meta,
    pos_vendor_api_meta_t* vendor_api_meta,
    pos_support_api_meta_t* support_api_meta
){
    pos_retval_t retval = POS_SUCCESS;
    uint64_t i;
    std::string api_snake_name;
    POSCodeGen_CppSourceFile *worker_file;
    POSCodeGen_CppBlock *wk_function_namespace, *api_namespace, *worker_function;
    std::string worker_file_directory;

    POS_CHECK_POINTER(vendor_api_meta);
    POS_CHECK_POINTER(support_api_meta);

    // for those APIs to be customized logic, we just omit
    if(support_api_meta->worker_type == std::string("customized")){
        goto exit;
    }

    api_snake_name = posautogen_utils_camel2snake(support_api_meta->name);

    worker_file_directory = this->worker_directory + std::string("/") + support_api_meta->library_name;
    if(!std::filesystem::exists(worker_file_directory)){
        try {
            std::filesystem::create_directory(worker_file_directory);
        } catch (const std::exception& e) {
            POS_WARN_C(
                "failed to create new library directory for the generated codes: worker_file_directory(%s)",
                worker_file_directory.c_str()
            );
            retval = POS_FAILED;
            goto exit;
        }
    }

    // create worker file
    worker_file = new POSCodeGen_CppSourceFile(
        worker_file_directory
        + std::string("/")
        + support_api_meta->name
        + std::string(".cpp")
    );
    POS_CHECK_POINTER(worker_file);
    
    // add basic headers to the worker file
    worker_file->add_preprocess("#include <iostream>");
    worker_file->add_preprocess("#include <cstdint>");
    worker_file->add_preprocess("#include \"pos/include/common.h\"");
    worker_file->add_preprocess("#include \"pos/include/client.h\"");
    for(i=0; i<support_api_meta->dependent_headers.size(); i++){
        worker_file->add_preprocess(std::format("#include <{}>", support_api_meta->dependent_headers[i]));
    }

    // create wk_function namespace
    wk_function_namespace = new POSCodeGen_CppBlock(
        /* field name */ "namespace wk_functions",
        /* need_braces */ true,
        /* foot_comment */ "namespace wk_functions"
    );
    POS_CHECK_POINTER(wk_function_namespace);
    worker_file->add_block(wk_function_namespace);

    // create api namespace
    retval = wk_function_namespace->allocate_block(
        /* field name */ std::string("namespace ") + api_snake_name,
        /* new_block */ &api_namespace,
        /* need_braces */ true,
        /* foot_comment */ std::string("namespace ") + api_snake_name,
        /* need_ended_semicolon */ false,
        /* level_offset */ 0
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to allocate cpp block for api namespace while generating worker function: "
            "api_name(%s)",
            api_snake_name
        );
    }
    POS_CHECK_POINTER(api_namespace);

    // create function POS_WK_FUNC_LAUNCH
    retval = api_namespace->allocate_block(
        /* field name */ std::string("POS_WK_FUNC_LAUNCH()"),
        /* new_block */ &worker_function,
        /* need_braces */ true,
        /* foot_comment */ "",
        /* need_ended_semicolon */ false,
        /* level_offset */ 1
    );
    if(unlikely(retval != POS_SUCCESS)){
        POS_WARN_C(
            "failed to allocate cpp block for POS_WK_FUNC_LAUNCH while generating worker function: "
            "api_name(%s)",
            api_snake_name
        );
    }
    POS_CHECK_POINTER(worker_function);

    if(unlikely(POS_SUCCESS != (
        retval = this->__insert_code_worker_for_target(
            vender_header_file_meta,
            support_header_file_meta,
            vendor_api_meta,
            support_api_meta,
            worker_file,
            wk_function_namespace,
            api_namespace,
            worker_function
        )
    ))){
        POS_WARN_C("failed to generate worker when inserted target-specific code");
        goto exit;
    }

    worker_file->archive();

exit:
    return retval;
}


pos_retval_t POSAutogener::__generate_auxiliary_files(
    std::vector<pos_vendor_header_file_meta*>& vendor_header_file_meta_list,
    std::vector<pos_support_api_meta_t*>& support_api_meta_list
){
    pos_retval_t retval = POS_SUCCESS;

    // reorder the APIs by their indices
    std::sort(
        support_api_meta_list.begin(),
        support_api_meta_list.end(),
        [](pos_support_api_meta_t* A, pos_support_api_meta_t* B) -> bool {
            return A->index < B->index;
        }
    );

    // generate api_index.h
    auto __generate_api_index_h = [&](){
        uint64_t i;
        POSCodeGen_CppSourceFile *api_index_h;
        pos_support_api_meta_t *api_meta;

        api_index_h = new POSCodeGen_CppSourceFile(
            this->gen_directory 
            + std::string("/")
            + std::string("api_index.h")
        );
        POS_CHECK_POINTER(api_index_h);
        
        api_index_h->add_preprocess("#pragma once");
        for(i=0; i<support_api_meta_list.size(); i++){
            POS_CHECK_POINTER(api_meta = support_api_meta_list[i]);
            api_index_h->add_preprocess(std::format(
                "#define PosApiIndex_{} {}",
                api_meta->name, api_meta->index
            ));
        }

        api_index_h->archive();
    };

    // generate api_context.h
    auto __generate_api_context_h = [&](){
        uint64_t i;
        pos_support_api_meta_t *api_meta;
        std::string target_uppercase;
        POSCodeGen_CppSourceFile *api_context_h = nullptr;
        POSCodeGen_CppBlock *class_POSApiManager_TARGET = nullptr;
        POSCodeGen_CppBlock *class_POSApiManager_TARGET_function_init = nullptr;
        POSCodeGen_CppBlock *class_POSApiManager_TARGET_function_declare_cast_pos_retval = nullptr;
        POSCodeGen_CppBlock *func_declare_pos_is_hijacked = nullptr;

        auto ____get_pos_api_type_string = [](pos_api_type_t api_type) -> std::string {
            switch (api_type)
            {
            case kPOS_API_Type_Create_Resource:
                return "kPOS_API_Type_Create_Resource";

            case kPOS_API_Type_Delete_Resource:
                return "kPOS_API_Type_Delete_Resource";

            case kPOS_API_Type_Get_Resource:
                return "kPOS_API_Type_Get_Resource";

            case kPOS_API_Type_Set_Resource:
                return "kPOS_API_Type_Set_Resource";
            
            default:
                POS_ERROR("this is a bug");
            }
        };


        api_context_h = new POSCodeGen_CppSourceFile(
            this->gen_directory
            + std::string("/")
            + std::string("api_context.h")
        );
        POS_CHECK_POINTER(api_context_h);

        // include headers
        api_context_h->add_preprocess("#pragma once");
        api_context_h->add_preprocess("#include <iostream>");
        api_context_h->add_preprocess("#include <vector>");
        for(i=0; i<vendor_header_file_meta_list.size(); i++){
            api_context_h->add_preprocess(std::format(
                "#include <{}>",
                vendor_header_file_meta_list[i]->file_name
            ));
        }
        api_context_h->add_preprocess("#include \"pos/include/common.h\"");
        api_context_h->add_preprocess("#include \"pos/include/api_context.h\"");
        api_context_h->add_preprocess(std::format(
            "#include \"pos/{}_impl/api_index.h\"", this->target
        ));

        // declare POSApiManager_TARGET class
        target_uppercase = this->target;
        std::transform(target_uppercase.begin(), target_uppercase.end(), target_uppercase.begin(), ::toupper);
        class_POSApiManager_TARGET = new POSCodeGen_CppBlock(
            /* field name */ std::format(
                "/*\n"
                " *  \\brief    API manager of target {}\n"
                " */\n"
                "class POSApiManager_{} : public POSApiManager",
                target_uppercase,
                target_uppercase
            ),
            /* need_braces */ true,
            /* foot_comment */ std::format("class POSApiManager_{}", target_uppercase),
            /* need_ended_semicolon */ true,
            /* level */ 0
        );
        POS_CHECK_POINTER(class_POSApiManager_TARGET);
        api_context_h->add_block(class_POSApiManager_TARGET);

        // define constructor and deconstructor for POSApiManager_TARGET
        class_POSApiManager_TARGET->append_content("public:", -3);
        class_POSApiManager_TARGET->append_content(std::format(
            "POSApiManager_{}(){{}}", target_uppercase
        ));
        class_POSApiManager_TARGET->append_content(std::format(
            "~POSApiManager_{}() = default;", target_uppercase
        ));

        // define POSApiManager_TARGET::init()
        retval = class_POSApiManager_TARGET->allocate_block(
            /* field name */ std::string("void init() override"),
            /* new_block */ &class_POSApiManager_TARGET_function_init,
            /* need_braces */ true
        );
        if(unlikely(retval != POS_SUCCESS)){
            POS_WARN_C(
                "failed to allocate cpp block for POSApiManager_TARGET::init() while generating api_context.h"
            );
            goto __exit;
        }
        POS_CHECK_POINTER(class_POSApiManager_TARGET_function_init);

        class_POSApiManager_TARGET_function_init->append_content("this->api_metas.insert({");
        for(i=0; i<support_api_meta_list.size(); i++){
            POS_CHECK_POINTER(api_meta = support_api_meta_list[i]);
            class_POSApiManager_TARGET_function_init->append_content(
                /* content */ std::format(
                    "{{\n"
                    "    /* api_id */ PosApiIndex_{},\n"
                    "    {{\n"
                    "        /* is_sync */  {},\n"
                    "        /* api_type */ {},\n"
                    "        /* api_name */ \"{}\",\n"
                    "    }}\n"
                    "}}{}"
                    ,
                    api_meta->name,
                    api_meta->is_sync ? "true" : "false",
                    ____get_pos_api_type_string(api_meta->api_type),
                    api_meta->name,
                    i < support_api_meta_list.size()-1 ? "," : ""
                ),
                /* char_offset */ 4
            );
        }
        class_POSApiManager_TARGET_function_init->append_content("});");

        // define POSApiManager_TARGET::cast_pos_retval(pos_retval_t)
        retval = class_POSApiManager_TARGET->allocate_block(
            /* field name */ std::string("int cast_pos_retval(pos_retval_t pos_retval) override"),
            /* new_block */ &class_POSApiManager_TARGET_function_declare_cast_pos_retval,
            /* need_braces */ true
        );
        POS_CHECK_POINTER(class_POSApiManager_TARGET_function_declare_cast_pos_retval);
        class_POSApiManager_TARGET_function_declare_cast_pos_retval->append_content(
            "if(pos_retval == POS_SUCCESS)\n"
            "   return 0;\n"
            "else\n"
            "   return -1;\n"
        );

        // declare pos_is_hijacked function
        func_declare_pos_is_hijacked = new POSCodeGen_CppBlock(
            /* field name */
                "/*\n"
                " *  \\brief    check whether specified API is hijacked by PhOS\n"
                " *  \\param    api_id    index of the API to be checked\n"
                " *  \\return   true if supported, false if unsupported\n"
                " */\n"
                "bool pos_is_hijacked(uint64_t api_id);"
            ,
            /* need_braces */ false,
            /* foot_comment */ "",
            /* need_ended_semicolon */ false,
            /* level */ 0
        );
        POS_CHECK_POINTER(func_declare_pos_is_hijacked);
        api_context_h->add_block(func_declare_pos_is_hijacked);

        api_context_h->archive();
    
    __exit:
    };

    // generate api_context.cpp
    auto __generate_api_context_cpp = [&](){
        uint64_t i = 0;
        std::string target_uppercase;
        pos_support_api_meta_t *api_meta = nullptr;
        POSCodeGen_CppSourceFile *api_context_cpp = nullptr;
        POSCodeGen_CppBlock *vector_pos_hijacked_cuda_apis = nullptr;
        POSCodeGen_CppBlock *function_cast_pos_retval = nullptr;
        POSCodeGen_CppBlock *function_pos_is_hijacked = nullptr;

        target_uppercase = this->target;
        std::transform(target_uppercase.begin(), target_uppercase.end(), target_uppercase.begin(), ::toupper);

        api_context_cpp = new POSCodeGen_CppSourceFile(
            this->gen_directory
            + std::string("/")
            + std::string("api_context.cu")
        );
        POS_CHECK_POINTER(api_context_cpp);

        // include headers
        api_context_cpp->add_preprocess("#include <iostream>");
        api_context_cpp->add_preprocess("#include <vector>");
        api_context_cpp->add_preprocess("#include <stdint.h>");
        api_context_cpp->add_preprocess("#include \"pos/include/common.h\"");
        api_context_cpp->add_preprocess("#include \"pos/include/api_context.h\"");
        api_context_cpp->add_preprocess(std::format(
            "#include \"pos/{}_impl/api_index.h\"", this->target
        ));
        api_context_cpp->add_preprocess(std::format(
            "#include \"pos/{}_impl/api_context.h\"", this->target
        ));

        // declare vector of indices of supported APIs
        vector_pos_hijacked_cuda_apis = new POSCodeGen_CppBlock(
            /* field name */ "",
            /* need_braces */ false,
            /* foot_comment */ "",
            /* need_ended_semicolon */ true,
            /* level */ 0
        );
        POS_CHECK_POINTER(vector_pos_hijacked_cuda_apis);
        api_context_cpp->add_block(vector_pos_hijacked_cuda_apis);

        // insert vector
        vector_pos_hijacked_cuda_apis->append_content(
            std::format("std::vector<uint64_t> pos_hijacked_{}_apis({{", this->target),
            -4
        );
        for(i=0; i<support_api_meta_list.size(); i++){
            POS_CHECK_POINTER(api_meta = support_api_meta_list[i]);
            vector_pos_hijacked_cuda_apis->append_content(
                std::format(
                    "PosApiIndex_{}{}",
                    api_meta->name,
                    i == support_api_meta_list.size()-1 ? "" : ","
                )
            );
        }
        vector_pos_hijacked_cuda_apis->append_content(
            "});", -4
        );

        // define POSApiManager_TARGET::cast_pos_retval
        // function_cast_pos_retval = new POSCodeGen_CppBlock(
        //     /* field name */ std::format("int POSApiManager_{}::cast_pos_retval(pos_retval_t pos_retval)", target_uppercase),
        //     /* need_braces */ true,
        //     /* foot_comment */ "",
        //     /* need_ended_semicolon */ false,
        //     /* level */ 0
        // );
        // POS_CHECK_POINTER(function_cast_pos_retval);
        // api_context_cpp->add_block(function_cast_pos_retval);
        // function_cast_pos_retval->append_content(clear
        //     "if(pos_retval == POS_SUCCESS)\n"
        //     "   return 0;\n"
        //     "else\n"
        //     "   return -1;\n"
        // );

        // define pos_is_hijacked function
        function_pos_is_hijacked = new POSCodeGen_CppBlock(
            /* field name */ "bool pos_is_hijacked(uint64_t api_id)",
            /* need_braces */ true,
            /* foot_comment */ "",
            /* need_ended_semicolon */ false,
            /* level */ 0
        );
        POS_CHECK_POINTER(function_pos_is_hijacked);
        api_context_cpp->add_block(function_pos_is_hijacked);
        function_pos_is_hijacked->append_content(
            "uint64_t i=0;\n"
            "for(i=0; i<pos_hijacked_cuda_apis.size(); i++){\n"
            "    if(unlikely(pos_hijacked_cuda_apis[i] == api_id)){\n"
            "        return true;\n"
            "    }\n"
            "}\n"
            "return false;"
        );

        api_context_cpp->archive();
    };


    // generate parser_functions.h
    auto __generate_parser_functions_h = [&](){
        pos_retval_t retval;
        uint64_t i;
        std::string target_uppercase, api_snake_name;
        pos_support_api_meta_t *api_meta;
        POSCodeGen_CppSourceFile *parser_functions_h;
        POSCodeGen_CppBlock *namespace_ps_functions;
        
        parser_functions_h = new POSCodeGen_CppSourceFile(
            this->gen_directory 
            + std::string("/")
            + std::string("parser_functions.h")
        );
        POS_CHECK_POINTER(parser_functions_h);

        // include headers
        parser_functions_h->add_preprocess("#pragma once");
        parser_functions_h->add_preprocess("#include \"pos/include/common.h\"");
        parser_functions_h->add_preprocess("#include \"pos/include/parser.h\"");
        parser_functions_h->add_preprocess("#include \"pos/include/api_context.h\"");
        parser_functions_h->add_preprocess(std::format(
            "#include \"pos/{}_impl/api_index.h\"", this->target
        ));

        // declare namespace ps_functions
        namespace_ps_functions = new POSCodeGen_CppBlock(
            /* field name */ "namespace ps_functions",
            /* need_braces */ true,
            /* foot_comment */ "namespace ps_functions",
            /* need_ended_semicolon */ false,
            /* level */ 0
        );
        POS_CHECK_POINTER(namespace_ps_functions);
        parser_functions_h->add_block(namespace_ps_functions);

        // declare parser functions inside namespace
        for(i=0; i<support_api_meta_list.size(); i++){
            POS_CHECK_POINTER(api_meta = support_api_meta_list[i]);
            namespace_ps_functions->append_content(
                std::format(
                    "POS_PS_DECLARE_FUNCTIONS({});",
                    posautogen_utils_camel2snake(api_meta->name)
                )
            );
        }

        parser_functions_h->archive();
    
    __exit:
    };

    // generate parser_functions.cpp
    auto __generate_parser_functions_cpp = [&](){
        pos_retval_t retval;
        uint64_t i;
        std::string target_uppercase, api_snake_name;
        pos_support_api_meta_t *api_meta;
        POSCodeGen_CppSourceFile *parser_functions_cpp;
        POSCodeGen_CppBlock *namespace_ps_functions;
        POSCodeGen_CppBlock *class_POSParser_TARGET_function_init_ps_functions;
        
        parser_functions_cpp = new POSCodeGen_CppSourceFile(
            this->gen_directory 
            + std::string("/")
            + std::string("parser_functions.cpp")
        );
        POS_CHECK_POINTER(parser_functions_cpp);

        // include headers
        parser_functions_cpp->add_preprocess("#pragma once");
        parser_functions_cpp->add_preprocess("#include \"pos/include/common.h\"");
        parser_functions_cpp->add_preprocess("#include \"pos/include/parser.h\"");
        parser_functions_cpp->add_preprocess("#include \"pos/include/api_context.h\"");
        parser_functions_cpp->add_preprocess(std::format(
            "#include \"pos/{}_impl/parser.h\"", this->target
        ));
        parser_functions_cpp->add_preprocess(std::format(
            "#include \"pos/{}_impl/api_index.h\"", this->target
        ));
        parser_functions_cpp->add_preprocess(std::format(
            "#include \"pos/{}_impl/parser_functions.h\"", this->target
        ));

        // declare function POSParser_TARGET::init_ps_functions
        target_uppercase = this->target;
        std::transform(target_uppercase.begin(), target_uppercase.end(), target_uppercase.begin(), ::toupper);
        class_POSParser_TARGET_function_init_ps_functions = new POSCodeGen_CppBlock(
            /* field name */ std::format(
                "pos_retval_t POSParser_{}::init_ps_functions()",
                target_uppercase
            ),
            /* need_braces */ true,
            /* foot_comment */ std::format(
                "pos_retval_t POSParser_{}::init_ps_functions",
                target_uppercase
            ),
            /* need_ended_semicolon */ false,
            /* level */ 0
        );
        POS_CHECK_POINTER(class_POSParser_TARGET_function_init_ps_functions);
        parser_functions_cpp->add_block(class_POSParser_TARGET_function_init_ps_functions);

        class_POSParser_TARGET_function_init_ps_functions->append_content("this->_parser_functions.insert({");
        for(i=0; i<support_api_meta_list.size(); i++){
            POS_CHECK_POINTER(api_meta = support_api_meta_list[i]);

            api_snake_name = posautogen_utils_camel2snake(api_meta->name);

            class_POSParser_TARGET_function_init_ps_functions->append_content(
                std::format(
                    "{{   PosApiIndex_{}, ps_functions::{}::parse }}{}",
                    api_meta->name,
                    api_snake_name,
                    i == support_api_meta_list.size()-1 ? "" : ","
                ),
                4
            );
        }
        class_POSParser_TARGET_function_init_ps_functions->append_content("});");
        class_POSParser_TARGET_function_init_ps_functions->append_content("return POS_SUCCESS;");

        parser_functions_cpp->archive();
    
    __exit:
    };


    // generate worker_functions.h
    auto __generate_worker_functions_h = [&](){
        pos_retval_t retval;
        uint64_t i;
        std::string target_uppercase, api_snake_name;
        pos_support_api_meta_t *api_meta;
        POSCodeGen_CppSourceFile *worker_functions_h;
        POSCodeGen_CppBlock *namespace_wk_functions;
        
        worker_functions_h = new POSCodeGen_CppSourceFile(
            this->gen_directory 
            + std::string("/")
            + std::string("worker_functions.h")
        );
        POS_CHECK_POINTER(worker_functions_h);

        // include headers
        worker_functions_h->add_preprocess("#pragma once");
        worker_functions_h->add_preprocess("#include \"pos/include/common.h\"");
        worker_functions_h->add_preprocess("#include \"pos/include/worker.h\"");
        worker_functions_h->add_preprocess("#include \"pos/include/api_context.h\"");
        worker_functions_h->add_preprocess(std::format(
            "#include \"pos/{}_impl/api_index.h\"", this->target
        ));

        // declare namespace ps_functions
        namespace_wk_functions = new POSCodeGen_CppBlock(
            /* field name */ "namespace wk_functions",
            /* need_braces */ true,
            /* foot_comment */ "namespace wk_functions",
            /* need_ended_semicolon */ false,
            /* level */ 0
        );
        POS_CHECK_POINTER(namespace_wk_functions);
        worker_functions_h->add_block(namespace_wk_functions);

        // declare parser functions inside namespace
        for(i=0; i<support_api_meta_list.size(); i++){
            POS_CHECK_POINTER(api_meta = support_api_meta_list[i]);
            namespace_wk_functions->append_content(
                std::format(
                    "POS_WK_DECLARE_FUNCTIONS({});",
                    posautogen_utils_camel2snake(api_meta->name)
                )
            );
        }

        worker_functions_h->archive();
    
    __exit:
    };


    // generate parser_functions.cpp
    auto __generate_worker_functions_cpp = [&](){
        pos_retval_t retval;
        uint64_t i;
        std::string target_uppercase, api_snake_name;
        pos_support_api_meta_t *api_meta;
        POSCodeGen_CppSourceFile *worker_functions_cpp;
        POSCodeGen_CppBlock *namespace_wk_functions;
        POSCodeGen_CppBlock *class_POSWorker_TARGET_function_init_wk_functions;
        
        worker_functions_cpp = new POSCodeGen_CppSourceFile(
            this->gen_directory 
            + std::string("/")
            + std::string("worker_functions.cpp")
        );
        POS_CHECK_POINTER(worker_functions_cpp);

        // include headers
        worker_functions_cpp->add_preprocess("#pragma once");
        worker_functions_cpp->add_preprocess("#include \"pos/include/common.h\"");
        worker_functions_cpp->add_preprocess("#include \"pos/include/worker.h\"");
        worker_functions_cpp->add_preprocess("#include \"pos/include/api_context.h\"");
        worker_functions_cpp->add_preprocess(std::format(
            "#include \"pos/{}_impl/api_index.h\"", this->target
        ));
        worker_functions_cpp->add_preprocess(std::format(
            "#include \"pos/{}_impl/worker.h\"", this->target
        ));
        worker_functions_cpp->add_preprocess(std::format(
            "#include \"pos/{}_impl/worker_functions.h\"", this->target
        ));

        // declare function POSParser_TARGET::init_wk_functions
        target_uppercase = this->target;
        std::transform(target_uppercase.begin(), target_uppercase.end(), target_uppercase.begin(), ::toupper);
        class_POSWorker_TARGET_function_init_wk_functions = new POSCodeGen_CppBlock(
            /* field name */ std::format(
                "pos_retval_t POSWorker_{}::init_wk_functions()",
                target_uppercase
            ),
            /* need_braces */ true,
            /* foot_comment */ std::format(
                "pos_retval_t POSWorker_{}::init_wk_functions",
                target_uppercase
            ),
            /* need_ended_semicolon */ false,
            /* level */ 0
        );
        POS_CHECK_POINTER(class_POSWorker_TARGET_function_init_wk_functions);
        worker_functions_cpp->add_block(class_POSWorker_TARGET_function_init_wk_functions);

        class_POSWorker_TARGET_function_init_wk_functions->append_content("this->_launch_functions.insert({");
        for(i=0; i<support_api_meta_list.size(); i++){
            POS_CHECK_POINTER(api_meta = support_api_meta_list[i]);

            api_snake_name = posautogen_utils_camel2snake(api_meta->name);

            class_POSWorker_TARGET_function_init_wk_functions->append_content(
                std::format(
                    "{{   PosApiIndex_{}, wk_functions::{}::launch }}{}",
                    api_meta->name,
                    api_snake_name,
                    i == support_api_meta_list.size()-1 ? "" : ","
                ),
                4
            );
        }
        class_POSWorker_TARGET_function_init_wk_functions->append_content("});");
        class_POSWorker_TARGET_function_init_wk_functions->append_content("return POS_SUCCESS;");

        worker_functions_cpp->archive();
    
    __exit:
    };

    __generate_api_index_h();
    __generate_api_context_h();
    __generate_api_context_cpp();
    __generate_parser_functions_h();
    __generate_parser_functions_cpp();
    __generate_worker_functions_h();
    __generate_worker_functions_cpp();

exit:
    return retval;
}
