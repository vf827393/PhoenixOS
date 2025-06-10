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

#include "pos/include/command.h"
#include "pos/include/log.h"
#include "pos/include/agent.h"

/*!
 *  \brief  create new agent
 *  \return pointer to the created agent
 */
extern "C" {


POSAgent* pos_create_agent();


/*!
 *  \brief  destory agent
 *  \param  pos_cuda_ws pointer to the agent to be destoried
 *  \return 0 for successfully destory
 *          1 for failed
 */
int pos_destory_agent(POSAgent* pos_agent);


/*!
 *  \brief  get agent uuid
 *  \param  pos_agent pointer to the agent to be queried
 *  \return uuid of the agent
 */
uint64_t pos_agent_get_uuid(POSAgent* pos_agent);


/*!
 *  \brief  query agent ready state
 *  \param  pos_agent pointer to the agent to be queried
 *  \return 0 for non-ready
 *          1 for ready
 */
int pos_query_agent_ready_state(POSAgent* pos_agent);


} // extern "C"
