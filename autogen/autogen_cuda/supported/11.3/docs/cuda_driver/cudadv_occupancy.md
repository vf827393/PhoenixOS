<h2>PhOS Support: CUDA 11.3 - Driver APIs - Occupancy (0/5)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__OCCUPANCY.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuOccupancyAvailableDynamicSMemPerBlock ( size_t* dynamicSmemSize, CUfunction func, int  numBlocks, int  blockSize )</code><br>
Returns dynamic shared memory available per block when launching numBlocks blocks on SM.
</td>
</tr>
<tr>
<td>1000</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize )</code><br>
Returns occupancy of a function.
</td>
</tr>
<tr>
<td>1001</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags )</code><br>
Returns occupancy of a function.
</td>
</tr>
<tr>
<td>1002</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuOccupancyMaxPotentialBlockSize ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit )</code><br>
Suggest a launch configuration with reasonable occupancy.
</td>
</tr>
<tr>
<td>1003</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuOccupancyMaxPotentialBlockSizeWithFlags ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit, unsigned int  flags )</code><br>
Suggest a launch configuration with reasonable occupancy.
</td>
</tr>
<tr>
<td>1004</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
