#pragma once

#include <iostream>
#include <stdint.h>

#include "pos/include/common.h"
#include "pos/include/log.h"

/*!
 *  \brief  represent an instance of a specific kind of physical resource on the XPU
 */
class POSResource {
 public:
    /*!
     *  \brief  constructor
     *  \param  adddr   physical address of the resource
     *  \param  size    size of resource state
     */
    POSResource(void *addr, size_t size) : _addr(addr), _size(size) {}
    
 protected:
    // physical address of the resource
    void *_addr;

    // size of resource state
    size_t _size;
};

/*!
 *  \brief  manage one kind of physical resource on the XPU
 */
template<class T_POSResource>
class POSResourceManager {
 public:
 private:
    /*!
     *  \brief      budget of this kind of resource on the physical platform
     *  \example    1. for POSResource_Device, this value represents #XPUs;
     *              2. for POSResource_Memory, this value represents overall device memory;
     *              3. 
     */
    uint64_t _budget;
    uint64_t _used;
    std::vector<T_POSResource*> _resources;
};
