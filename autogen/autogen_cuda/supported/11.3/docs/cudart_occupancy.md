<h2>PhOS Support: CUDA 11.3 - Runtime APIs - Occupancy (0/3)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__OCCUPANCY.html#group__CUDART__OCCUPANCY

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock ( size_t* dynamicSmemSize, const void* func, int  numBlocks, int  blockSize )</code><br>
Returns dynamic shared memory available per block when launching numBlocks blocks on SM.
</td>
</tr>
<tr>
<td>350</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t  cudaOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize )</code><br>
Returns occupancy for a device function.
</td>
</tr>
<tr>
<td>351</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags )</code><br>
Returns occupancy for a device function with the specified flags.
</td>
</tr>
<tr>
<td>352</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
