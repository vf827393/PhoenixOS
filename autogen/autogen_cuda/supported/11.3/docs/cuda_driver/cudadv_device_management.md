<h2>PhOS Support: CUDA 11.3 - Driver APIs - Device Management (0/12)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGet ( CUdevice* device, int ordinal )</code><br>
Returns a handle to a compute device.
</td>
</tr>
<tr>
<td>650</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetAttribute ( int* pi, CUdevice_attribute attrib, CUdevice dev )</code><br>
Returns information about the device.
</td>
</tr>
<tr>
<td>651</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetCount ( int* count )</code><br>
Returns the number of compute-capable devices.
</td>
</tr>
<tr>
<td>652</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetDefaultMemPool ( CUmemoryPool* pool_out, CUdevice dev )</code><br>
Returns the default mempool of a device.
</td>
</tr>
<tr>
<td>653</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetLuid ( char* luid, unsigned int* deviceNodeMask, CUdevice dev )</code><br>
Return an LUID and device node mask for the device.
</td>
</tr>
<tr>
<td>654</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetMemPool ( CUmemoryPool* pool, CUdevice dev )</code><br>
Gets the current mempool for a device.
</td>
</tr>
<tr>
<td>655</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetName ( char* name, int len, CUdevice dev )</code><br>
Returns an identifier string for the device.
</td>
</tr>
<tr>
<td>656</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetNvSciSyncAttributes ( void* nvSciSyncAttrList, CUdevice dev, int flags )</code><br>
Return NvSciSync attributes that this device can support.
</td>
</tr>
<tr>
<td>657</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetTexture1DLinearMaxWidth ( size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev )</code><br>
Returns the maximum number of elements allocatable in a 1D linear texture for a given texture element size.
</td>
</tr>
<tr>
<td>658</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceGetUuid ( CUuuid* uuid, CUdevice dev )</code><br>
Return an UUID for the device.
</td>
</tr>
<tr>
<td>659</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceSetMemPool ( CUdevice dev, CUmemoryPool pool )</code><br>
Sets the current memory pool of a device.
</td>
</tr>
<tr>
<td>660</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDeviceTotalMem ( size_t* bytes, CUdevice dev )</code><br>
Returns the total amount of memory on the device.
</td>
</tr>
<tr>
<td>661</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
