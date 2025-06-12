<h2>PhOS Support: CUDA 11.3 - Driver APIs - Stream Memory Operations (0/5)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__MEMOP.html#group__CUDA__MEMOP

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamBatchMemOp ( CUstream stream, unsigned int  count, CUstreamBatchMemOpParams* paramArray, unsigned int  flags )</code><br>
Batch operations to synchronize the stream via memory operations.
</td>
</tr>
<tr>
<td>900</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamWaitValue32 ( CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags )</code><br>
Wait on a memory location.
</td>
</tr>
<tr>
<td>901</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamWaitValue64 ( CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags )</code><br>
Wait on a memory location.
</td>
</tr>
<tr>
<td>902</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamWriteValue32 ( CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags )</code><br>
Write a value to memory.
</td>
</tr>
<tr>
<td>903</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamWriteValue64 ( CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags )</code><br>
Write a value to memory.
</td>
</tr>
<tr>
<td>904</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
