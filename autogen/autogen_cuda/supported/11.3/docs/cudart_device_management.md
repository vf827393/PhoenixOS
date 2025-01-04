<h2>PhOS Support: CUDA Runtime APIS - Device Management (0/32)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaChooseDevice ( int* device, const cudaDeviceProp* prop )</code><br>
Select compute-device which best matches criteria.
</td>
</tr>

<tr>
<td>100</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceFlushGPUDirectRDMAWrites ( cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope )</code><br>
Blocks until remote writes are visible to the specified scope.
</td>
</tr>

<tr>
<td>101</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t  cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device )</code><br>
Returns information about the device.
</td>
</tr>

<tr>
<td>102</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceGetByPCIBusId ( int* device, const char* pciBusId )</code><br>
Returns a handle to a compute device.
</td>
</tr>
<tr>
<td>103</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t  cudaDeviceGetCacheConfig ( cudaFuncCache ** pCacheConfig )</code><br>
Returns the preferred cache configuration for the current device.
</td>
</tr>
<tr>
<td>104</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceGetDefaultMemPool ( cudaMemPool_t* memPool, int  device )</code><br>
Returns the default mempool of a device.
</td>
</tr>
<tr>
<td>105</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=3>
<code>cudaError_t  cudaDeviceGetLimit ( size_t* pValue, cudaLimit limit )</code><br>
Returns resource limits.
</td>
</tr>
<tr>
<td>106</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceGetMemPool ( cudaMemPool_t* memPool, int  device )</code><br>
Gets the current mempool for a device.
</td>
</tr>
<tr>
<td>107</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceGetNvSciSyncAttributes ( void* nvSciSyncAttrList, int  device, int  flags )</code><br>
Return NvSciSync attributes that this device can support.
</td>
</tr>
<tr>
<td>108</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceGetP2PAttribute ( int* value, cudaDeviceP2PAttr attr, int  srcDevice, int  dstDevice )</code><br>
Queries attributes of the link between two devices.
</td>
</tr>
<tr>
<td>109</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceGetPCIBusId ( char* pciBusId, int  len, int  device )</code><br>
Returns a PCI Bus Id string for the device.
</td>
</tr>
<tr>
<td>110</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t  cudaDeviceGetSharedMemConfig ( cudaSharedMemConfig ** pConfig )</code><br>
Returns the shared memory configuration for the current device.
</td>
</tr>
<tr>
<td>111</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceGetStreamPriorityRange ( int* leastPriority, int* greatestPriority )</code><br>
Returns numerical values that correspond to the least and greatest stream priorities.
</td>
</tr>
<tr>
<td>112</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceGetTexture1DLinearMaxWidth ( size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int  device )</code><br>
Returns the maximum number of elements allocatable in a 1D linear texture for a given element size.
</td>
</tr>
<tr>
<td>113</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceReset ( void )</code><br>
Destroy all allocations and reset all state on the current device in the current process.
</td>
</tr>
<tr>
<td>114</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceSetCacheConfig ( cudaFuncCache cacheConfig )</code><br>
Sets the preferred cache configuration for the current device.
</code_to_rewrite>
</td>
</tr>
<tr>
<td>115</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceSetLimit ( cudaLimit limit, size_t value )</code><br>
Set resource limits.
</td>
</tr>
<tr>
<td>116</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceSetMemPool ( int  device, cudaMemPool_t memPool )</code><br>
Sets the current memory pool of a device.
</td>
</tr>
<tr>
<td>117</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceSetSharedMemConfig ( cudaSharedMemConfig config )</code><br>
Sets the shared memory configuration for the current device.
</td>
</tr>
<tr>
<td>118</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t  cudaDeviceSynchronize ( void )</code><br>
Wait for compute device to finish.
</td>
</tr>
<tr>
<td>119</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr style="color:green;">
<td colspan=3>
<code>cudaError_t  cudaGetDevice ( int* device )</code><br>
Returns which device is currently being used.
</td>
</tr>
<tr style="color:green;">
<td>120</td>
<td>✓</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t  cudaGetDeviceCount ( int* count )</code><br>
Returns the number of compute-capable devices.
</td>
</tr>
<tr>
<td>121</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGetDeviceFlags ( unsigned int* flags )</code><br>
Gets the flags for the current device.
</td>
</tr>
<tr>
<td>122</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device )</code><br>
Returns information about the compute-device.
</td>
</tr>
<tr>
<td>123</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaIpcCloseMemHandle ( void* devPtr )</code><br>
Attempts to close memory mapped with cudaIpcOpenMemHandle.
</td>
</tr>
<tr>
<td>124</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaIpcGetEventHandle ( cudaIpcEventHandle_t* handle, cudaEvent_t event )</code><br>
Gets an interprocess handle for a previously allocated event.
</td>
</tr>
<tr>
<td>125</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr )</code><br>
Gets an interprocess memory handle for an existing device memory allocation.
</td>
</tr>
<tr>
<td>126</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaIpcOpenEventHandle ( cudaEvent_t* event, cudaIpcEventHandle_t handle )</code><br>
Opens an interprocess event handle for use in the current process.
</td>
</tr>
<tr>
<td>127</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags )</code><br>
Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
</td>
</tr>
<tr>
<td>128</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaSetDevice ( int  device )</code><br>
Set device to be used for GPU executions.
</td>
</tr>
<tr>
<td>129</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaSetDeviceFlags ( unsigned int  flags )</code><br>
Sets flags to be used for device executions.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=3>
<code>cudaError_t cudaSetValidDevices ( int* device_arr, int  len )</code><br>
Set a list of devices that can be used for CUDA.
</td>
</tr>
<tr>
<td>130</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
