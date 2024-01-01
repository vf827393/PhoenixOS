#pragma once

#include <iostream>
#include <string>
#include <map>

#include <stdint.h>

#include "pos/include/common.h"
#include "pos/include/log.h"

#define POS_ENABLE_SERVICE_TIRPC 1

/*!
 *  \brief  frontend transport type
 */
enum pos_service_typeid_t {
    kPOS_ServiceTypeId_Unknown=0,
    kPOS_ServiceTypeId_TIRPC
};

/*!
 *  \brief  transport frontend
 */
class POSService {
 public:
    POSService(){}
    ~POSService() = default;

    /*!
     *  \brief  accept the request according to the service context, update
     *          transport metadata and return the uuid of the remote client
     *  \param  svc service context
     *  \return the uuid of the remote client
     */
    virtual inline pos_client_uuid_t accept_req(void* svc) = 0;

    /*!
     *  \brief      obtain the information of the remote client 
     *              endpoint, in the form of string
     *  \example    [protocol:ip:port] of the remote client
     *  \param      svc service context
     *  \return     the obtained string
     */
    virtual inline std::string get_client_info_str(void* svc);

    // transport type id
    static const pos_service_typeid_t _type_id = kPOS_ServiceTypeId_Unknown;

 protected:
    // number of total request for each client during the runtime
    std::map<pos_client_uuid_t, uint64_t> _nb_reqs;
};


/* =================== Implementation: TIRPC-based Service =================== */
#if POS_ENABLE_SERVICE_TIRPC

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "tirpc/rpc/types.h"

/*!
 *  \brief  transport frontend of libtirpc
 */
class POSService_TIRPC : public POSService {
    /*!
     *  \brief  accept the request according to the service context, update
     *          transport metadata and return the uuid of the remote client
     *  \param  svc service context
     *  \return the uuid of the remote client
     */
    inline pos_client_uuid_t accept_req(void* ctx){
        struct svc_req *rqstp = (struct svc_req*)ctx;

        // we use the address of server-side transport handle as 
        // the uuid of this client
        pos_client_uuid_t retval = (pos_client_uuid_t)(rqstp->rq_xprt);

        if(likely(_nb_reqs[retval] < UINT_64_MAX)){ _nb_reqs[retval] += 1; }

        return (pos_client_uuid_t)(rqstp->rq_xprt);
    }

    /*!
     *  \brief      obtain the information of the remote client 
     *              endpoint, in the form of string
     *  \example    [protocol:ip:port] of the remote client
     *  \param      svc service context
     *  \return     the obtained string
     */
    inline std::string get_client_info_str(void* svc){
        struct svc_req *rqstp = (struct svc_req*)svc;
        char ip_str[128] = {0}, ret_str[256] = {0};
        
        inet_ntop(
            rqstp->rq_xprt->xp_raddr.sin6_family,
            &(rqstp->rq_xprt->xp_raddr.sin6_addr),
            ip_str, sizeof(ip_str)
        );

        sprintf(
            ret_str, "%s://%s:%u",
            /* protocol */ rqstp->rq_xprt->xp_netid,
            /* ip addr */ ip_str,
            /* port */ rqstp->rq_xprt->xp_raddr.sin6_port
        );

        return std::string(static_cast<const char*>(ret_str));
    }

    // transport type id
    static const pos_service_typeid_t _type_id = kPOS_ServiceTypeId_TIRPC;

 private:
};

#endif // POS_ENABLE_SERVICE_TIRPC
/* =============== End of Implementation: TIRPC-based Service ================ */
