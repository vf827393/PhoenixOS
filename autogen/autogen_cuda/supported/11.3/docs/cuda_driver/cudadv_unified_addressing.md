<h2>PhOS Support: CUDA 11.3 - Driver APIs - Unified Addressing (0/7)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__UNIFIED.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemAdvise ( CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device )</code><br>
Advise about the usage of a given memory range.
</td>
</tr>
<tr>
<td>840</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemPrefetchAsync ( CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream )</code><br>
Prefetches memory to the specified destination device.
</td>
</tr>
<tr>
<td>841</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemRangeGetAttribute ( void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count )</code><br>
Query an attribute of a given memory range.
</td>
</tr>
<tr>
<td>842</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuMemRangeGetAttributes ( void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count )</code><br>
Query attributes of a given memory range.
</td>
</tr>
<tr>
<td>843</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuPointerGetAttribute ( void* data, CUpointer_attribute attribute, CUdeviceptr ptr )</code><br>
Returns information about a pointer.
</td>
</tr>
<tr>
<td>844</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuPointerGetAttributes ( unsigned int  numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr )</code><br>
Returns information about a pointer.
</td>
</tr>
<tr>
<td>845</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuPointerSetAttribute ( const void* value, CUpointer_attribute attribute, CUdeviceptr ptr )</code><br>
Set attributes on a previously allocated memory region.
</td>
</tr>
<tr>
<td>846</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
