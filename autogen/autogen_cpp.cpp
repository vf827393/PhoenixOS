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
#include <cmath>

#include "autogen_common.h"


void POSCodeGen_CppBlock::archive(){
    uint64_t i;
    uint64_t content_ptr, block_ptr;
    uint8_t position_id;
    std::string line;
    std::pair<uint64_t, std::string> backspace_pair;

    auto __insert_space = [&](uint16_t num){
        uint16_t k; for(k=0; k<num; k++){ this->archived += " "; }
    };

    auto __write_line = [&](std::string& line){
        this->archived += line + std::string("\n");
    };

    auto __write_line_without_change_line = [&](std::string& line){
        this->archived += line;
    };

    auto __insert_empty_line = [&](uint64_t num){
        uint64_t l;
        for(l=0; l<num; l++){ this->archived += std::string("\n"); }   
    };

    auto __calculate_backspace_and_delete = [](std::string str) -> std::pair<uint64_t, std::string> {
        uint64_t count = 0;
        while (str.size() > 0 && str[0] == '\b') {
            count++; str = str.substr(1);
        }
        return std::pair<uint64_t, std::string>(count, str);
    };

    // block name
    if(this->_block_name.size() > 0){
        __insert_space(this->_level*4);
        __write_line(this->_block_name);
    }

    // brace start
    if(this->_need_braces == true){
        __insert_space(this->_level*4);
        std::string l_brace("{");
        __write_line(l_brace);
    }

    // var declarations
    for(i=0; i<this->_vars.size(); i++){
        __insert_space((this->_level+1)*4);
        __write_line(this->_vars[i]);
    }
    if(
        this->_vars.size() > 0
        && (this->_contents.size() > 0 or this->_inner_blocks.size() > 0)
    )
        __insert_empty_line(1);

    // insert content and block, within their insertion order
    content_ptr = 0;
    block_ptr = 0;
    while(!this->_insertion_position_q.empty()){
        position_id = this->_insertion_position_q.front();
        this->_insertion_position_q.pop();
        POS_ASSERT(position_id == 0 || position_id == 1);
        if(position_id == 0){
            // spliting contains based on '\n'
            std::stringstream ss(this->_contents[content_ptr]);
            while (std::getline(ss, line)) {
                backspace_pair = __calculate_backspace_and_delete(line);
                __insert_space((this->_level+1)*4 - backspace_pair.first);
                __write_line(backspace_pair.second);
            }
            content_ptr += 1;
            POS_ASSERT(content_ptr <= this->_contents.size());
        } else { // position_id == 1
            this->_inner_blocks[block_ptr]->archive();
            __write_line(this->_inner_blocks[block_ptr]->archived);
            block_ptr += 1;
            POS_ASSERT(block_ptr <= this->_inner_blocks.size());
        }

        if(block_ptr != this->_inner_blocks.size() || content_ptr != this->_contents.size()){
            __insert_empty_line(1);
        }
    }
    POS_ASSERT(content_ptr == this->_contents.size());
    POS_ASSERT(block_ptr == this->_inner_blocks.size());

    // brace stop
    if(this->_need_braces == true){
        __insert_space(this->_level*4);

        if(this->_need_ended_semicolon){
            std::string r_brace = std::string("};");
            if(this->_foot_comment.size() > 0)
                __write_line_without_change_line(r_brace);
            else
                __write_line(r_brace);
        } else {
            std::string r_brace = std::string("}");
            if(this->_foot_comment.size() > 0)
                __write_line_without_change_line(r_brace);
            else
                __write_line(r_brace);
        }

        if(this->_foot_comment.size() > 0){
            std::string foot_comment = std::string(" // ") + this->_foot_comment;
            __write_line(foot_comment);
        }
    }

exit:
    ;
}


bool POSCodeGen_CppBlock::declare_var(std::string var){ 
    // avoid duplication
    for(std::string& _var : this->_vars){
        if(unlikely(_var == var)){ return true; }
    }
    this->_vars.push_back(var);
    return false;
}


void POSCodeGen_CppBlock::append_content(std::string content, int64_t char_offset){
    uint16_t k;
    std::istringstream iss(content);
    std::ostringstream oss;
    std::string line;

    while (std::getline(iss, line)) {
        if(char_offset > 0)
            for(k=0; k<char_offset; k++){ oss << " "; }
        else
            for(k=0; k<std::abs(char_offset); k++){ oss << "\b"; }
        oss << line << "\n";
    }

    this->_contents.push_back(oss.str());
    this->_insertion_position_q.push(0);
}


pos_retval_t POSCodeGen_CppBlock::allocate_block(
    std::string block_name,
    POSCodeGen_CppBlock** new_block,
    bool need_braces,
    std::string foot_comment,
    bool need_ended_semicolon,
    int level_offset
){
    pos_retval_t retval = POS_SUCCESS;
    uint8_t new_level;

    POS_CHECK_POINTER(new_block);

    if(level_offset < 0){
        POS_ASSERT(std::abs(level_offset) <= static_cast<int>(this->_level));
    }
    
    if(level_offset >= 0){
        new_level = static_cast<uint8_t>(level_offset) + this->_level;
    } else {
        new_level = this->_level - static_cast<uint8_t>(std::abs(level_offset));
    }
    
    *new_block = new POSCodeGen_CppBlock(
        block_name, need_braces, foot_comment, need_ended_semicolon, new_level
    );
    POS_CHECK_POINTER(*new_block);

    this->_inner_blocks.push_back(*new_block);
    this->_insertion_position_q.push(1);

    return retval;
}


void POSCodeGen_CppSourceFile::add_preprocess(std::string include){
    this->_preprocess.push_back(include);
}


void POSCodeGen_CppSourceFile::add_block(POSCodeGen_CppBlock* block){
    POS_CHECK_POINTER(block);
    this->_blocks.push_back(block);
}


void POSCodeGen_CppSourceFile::archive(){
    uint64_t i;
    static std::string autogen_declare 
        = "/* ====== This File is Generated by PhOS Autogen Engine ====== */";

    auto __write_block = [&](std::string& block, bool change_line=true){
        if(change_line)
            this->archived += block + std::string("\n");
        else
            this->archived += block;
    };

    auto __insert_empty_line = [&](uint64_t num){
        uint64_t l;
        for(l=0; l<num; l++){ this->archived += std::string("\n"); }   
    };

    __write_block(autogen_declare);
    __insert_empty_line(1);

    for(i=0; i<this->_preprocess.size(); i++){ 
        __write_block(this->_preprocess[i]);
    }
    if(this->_preprocess.size() > 0 && this->_blocks.size() > 0)
        __insert_empty_line(2);

    for(i=0; i<this->_blocks.size(); i++){
        POS_CHECK_POINTER(this->_blocks[i]);
        this->_blocks[i]->archive();
        if(i == this->_blocks.size()-1){
            __write_block(this->_blocks[i]->archived, false);
        } else {
            __write_block(this->_blocks[i]->archived, false);
            __insert_empty_line(2);
        }
    }

    this->_file_stream << this->archived;
    this->_file_stream.flush();
}
