#pragma once

#include <iostream>

#include "pos/common.h"
#include "pos/transport.h"
#include "pos/oob.h"
#include "pos/api_context.h"

/*!
 *  \brief  client-side PhoenixOS agent, manages all POS resources
 *  \tparam transport implementation    
 */
template<class T_POSTransport>
class POSAgent {
 public:
    /*!
     *  \brief  constructor
     */
    POSAgent(){
        _pos_oob_client = new POSOobClient<T_POSTransport>(
            /* agent */ this,
            /* local_port */ POS_OOB_CLIENT_DEFAULT_PORT
        );
        POS_CHECK_POINTER(_pos_oob_client);

        // step 1: register client
        if(POS_SUCCESS != _pos_oob_client->call(kPOS_Oob_Register_Client, nullptr)){
            POS_ERROR_C_DETAIL("failed to register the client");
        }
        POS_DEBUG_C("successfully register client: uuid(%lu)", _uuid);

        // step 2: open transport (top-half)
        transport = new T_POSTransport(
            /* id*/ _uuid,
            /* non_blocking */ true,
            /* role */ kPOS_Transport_RoleId_Client,
            /* timeout */ 5000
        );
        POS_CHECK_POINTER(_pos_oob_client);
        POS_DEBUG_C("successfully open transport (top-half)");

        // step 3: connect transport
        if(POS_SUCCESS != _pos_oob_client->call(kPOS_Oob_Connect_Transport, nullptr)){
            POS_ERROR_C_DETAIL("failed to connect the transport");
        }
        POS_DEBUG_C("successfully connect the transport");

        // step 4: open transport (top-half)
        if(POS_SUCCESS != transport->init_bh()){
            POS_ERROR_C_DETAIL("failed to open transport (bottom-half)");
        }
        POS_DEBUG_C("successfully open transport (bottom-half)");
    }

    /*!
     *  \brief  deconstructor
     */
    ~POSAgent(){
        if(POS_SUCCESS != _pos_oob_client->call(kPOS_Oob_Unregister_Client, nullptr)){
            POS_ERROR_C_DETAIL("failed to unregister the client");
        }
        delete transport;
        delete _pos_oob_client;
    }

    /*!
     *  \brief  call the out-of-band function
     *  \param  id      the out-of-band function id
     *  \param  data    payload to call the function
     *  \return according to different function definitions
     */
    inline pos_retval_t oob_call(pos_oob_msg_typeid_t id, void* data){
        POS_CHECK_POINTER(data);
        return _pos_oob_client->call(id, data);
    }

    /*!
     *  \brief  PhoenixOS API call proxy
     *  \param  api_id  index of the called API
     *  \param  params  list of parameters of the called API
     */
    pos_retval_t api_call(uint64_t api_id, std::vector<POSAPIParamDesp_t> params);

    /*!
     *  \brief  set the uuid of the client
     *  \note   this function is invoked during the registeration process 
     *          (i.e., register_client oob type)
     */
    inline void set_uuid(pos_client_uuid_t id){ _uuid = id; }

    // pointer to the transportation layer
    T_POSTransport *transport;

 private:
    // pointer to the out-of-band client
    POSOobClient<T_POSTransport> *_pos_oob_client;

    // uuid of the client
    pos_client_uuid_t _uuid;
};

extern POSAgent<POSTransport_SHM> *pos_agent;
