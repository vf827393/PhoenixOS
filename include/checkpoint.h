#pragma once

#include <iostream>
#include <vector>

#include <stdint.h>

#include <pos/common.h>
#include <pos/log.h>

class POSCheckpointSlot {
 public:
    POSCheckpointSlot(uint64_t state_size) : _state_size(state_size) {
        if(likely(state_size > 0)){
            POS_CHECK_POINTER(_data = malloc(state_size));
        } else {
            POS_WARN_C("try to create checkpoint slot with state size 0, this is a bug");
            _data = nullptr;
        }
    }

    inline void* expose_pointer(){ return _data; }

    ~POSCheckpointSlot(){
        if(unlikely(_data == nullptr && _state_size > 0)){
            free(_data);
        }
    }

 private:
    uint64_t _state_size;
    void *_data;
};
using POSCheckpointSlot_ptr = std::shared_ptr<POSCheckpointSlot>;


class POSCheckpointBag {
 public:
    POSCheckpointBag(uint64_t state_size) : _state_size(state_size) {}
    ~POSCheckpointBag() = default;

    pos_retval_t apply_new_checkpoint(uint64_t version, POSCheckpointSlot_ptr* ptr){
        pos_retval_t retval = POS_SUCCESS;

        POS_CHECK_POINTER(ptr);

        if(unlikely(_ckpts.count(version) > 0)){
            POS_WARN_C("failed to apply checkpoint from bag, duplicated version: version(%lu)", version);
            retval = POS_FAILED_ALREADY_EXIST;
            goto exit;
        }

        *ptr = std::make_shared<POSCheckpointSlot>(_state_size);
        POS_CHECK_POINTER((*ptr).get());

        _ckpts[version] = (*ptr);
    
    exit:
        return retval;
    }

    /*!
     *  \brief  obtain overall memory consumption of this checkpoint bag
     *  \return overall memory consumption of this checkpoint bag
     */
    inline uint64_t get_memory_consumption(){
        return _state_size * _ckpts.size();
    }

    /*!
     *  \brief  clear all old checkpoint datas except the specified version (the latest one)
     *  \param  except_version  the specified version to be excluded from the deleting versions
     *  \return overall clear size
     */
    inline uint64_t clear_all_old_checkpoints(uint64_t except_version){
        uint64_t clear_size = 0;

        typename std::map<uint64_t, POSCheckpointSlot_ptr>::iterator iter;
        for(iter=_ckpts.begin(); iter!=_ckpts.end(); iter++){
            if(unlikely(iter->first == except_version)){
                continue;
            }
            POS_CHECK_POINTER(iter->second.get());
            iter = _ckpts.erase(iter);
            clear_size += _state_size;
        }

        return clear_size;
    }

    pos_retval_t get_checkpoint_by_version(uint64_t version, POSCheckpointSlot_ptr* ptr){
        pos_retval_t retval = POS_SUCCESS;

        POS_CHECK_POINTER(ptr);

        if(unlikely(_ckpts.count(version) == 0)){
            POS_WARN_C("failed to get checkpoint from bag, no version exist: version(%lu)", version);
            retval = POS_FAILED_NOT_EXIST;
            goto exit;
        }

        *ptr = _ckpts[version];

    exit:
        return retval;
    }

 private:
    // version -> checkpoint
    std::map<uint64_t, POSCheckpointSlot_ptr> _ckpts;
    uint64_t _state_size;
};

class POSHandle;
using POSHandle_ptr = std::shared_ptr<POSHandle>;
