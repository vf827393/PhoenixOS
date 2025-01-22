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
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <cstring>

#include "utils.h"

#include "pos/include/common.h"
#include "pos/include/log.h"


/*!
 *  \brief  contains a C++ block's content
 */
class POSCodeGen_CppBlock {
 public:
    /*!
     *  \brief  constructor
     *  \param  block_name              name of this block
     *  \param  need_braces             whether it needs braces to wrap this block
     *  \param  foot_comment            foot comment of this block
     *  \param  need_ended_semicolon    whether it needs a ended semicolon (;)
     *  \param  level                   level of this block
     */
    POSCodeGen_CppBlock(
        std::string block_name,
        bool need_braces=true,
        std::string foot_comment="",
        bool need_ended_semicolon=false,
        uint8_t level=0
    )   :   _block_name(block_name),
            _need_braces(need_braces),
            _foot_comment(foot_comment),
            _need_ended_semicolon(need_ended_semicolon),
            _level(level),
            archived(""){}
    ~POSCodeGen_CppBlock() = default;

    // archived this block after (after all blocks are inserted)
    std::string archived;

    /*!
     *  \brief  archive this block (after all blocks are inserted)
     */
    void archive();

    /*!
     *  \brief  declare new variable in this block
     *  \param  var the newly add variable
     *  \return bool mark whether the variable to be registered is duplicated
     */
    bool declare_var(std::string var);

    /*!
     *  \brief  allocate new inner block of this block
     *  \param  block_name              name of the new block
     *  \param  new_block               pointer to the newly created block
     *  \param  need_braces             whether it needs braces to wrap this block
     *  \param  foot_comment            foot comment of this block
     *  \param  need_ended_semicolon    whether it needs a ended semicolon (;)
     *  \param  level_offset            offset of the new block based on current block's level
     *  \return POS_SUCCESS for succesfully generation
     */
    pos_retval_t allocate_block(
        std::string block_name,
        POSCodeGen_CppBlock** new_block,
        bool need_braces=true,
        std::string foot_comment="",
        bool need_ended_semicolon=false,
        int level_offset=1
    );

    /*!
     *  \brief  append content to this block
     *  \note   only leaf node can have content
     *  \param  content         the content to be appended
     *  \param  char_offset     the offset (in char) of the content, based on the block's offset
     */
    void append_content(std::string content, int64_t char_offset=0);

 private:
    // insertion idx queue
    // this queue control the order of inserted content and block
    // value = 0: content, value = 1: block
    std::queue<uint8_t> _insertion_position_q;

    // all variables declarations in this block
    std::vector<std::string> _vars;

    // other inner block in this block
    std::vector<POSCodeGen_CppBlock*> _inner_blocks;

    // content of this block
    // only leaf node can have content
    std::vector<std::string> _contents;

    // name of this block
    std::string _block_name;

    // mark whetehr this block need braces to wrap
    bool _need_braces;

    // mark whether this block need footnote comment
    std::string _foot_comment;

    // mark whether it needs a ended semicolon (;)
    bool _need_ended_semicolon;

    // level of this block
    uint8_t _level;
};


/*! 
 *  \brief  contains a C++ source file's content
 */
class POSCodeGen_CppSourceFile {
 public:
    /*!
     *  \brief  constructor
     *  \param  file_path   path to the generated file
     */
    POSCodeGen_CppSourceFile(std::string file_path){
        this->_file_stream.open(file_path, std::ios::trunc);
        if(unlikely(!this->_file_stream)){
            POS_WARN_C("failed to create new file: path(%s)", file_path.c_str());
        }
    }
    ~POSCodeGen_CppSourceFile(){
        if(likely(this->_file_stream)){
            this->_file_stream.close();
        }
    }

    // archived this file after all content generated
    std::string archived;

    /*!
     *  \brief  declare new include in this block
     *  \param  var the newly add include
     */
    void add_preprocess(std::string include);

    /*!
     *  \brief  add a new block to this block
     *  \param  block   the newly add block
     */
    void add_block(POSCodeGen_CppBlock* block);
    
    /*!
     *  \brief  archive and write to the file (after all blocks are inserted)
     */
    void archive();

 private:
    // file stream to the output file
    std::ofstream _file_stream;

    // all include headers of this source file
    std::vector<std::string> _preprocess;

    // all blocks of this source file
    std::vector<POSCodeGen_CppBlock*> _blocks;
};
