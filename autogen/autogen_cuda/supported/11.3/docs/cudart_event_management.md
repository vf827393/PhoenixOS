# [Event Management (0/8)](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT)

<table>
<tr>
<th>Supported</th>
<th>Test Passed</th>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaEventCreate ( cudaEvent_t* event )</code><br>
Creates an event object.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaEventCreateWithFlags ( cudaEvent_t* event, unsigned int  flags )</code><br>
Creates an event object with the specified flags.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaEventDestroy ( cudaEvent_t event )</code><br>
Destroys an event object.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start, cudaEvent_t end )</code><br>
Computes the elapsed time between events.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaEventQuery ( cudaEvent_t event )</code><br>
Queries an event's status.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t  cudaEventRecord ( cudaEvent_t event, cudaStream_t stream = 0 )</code><br>
Records an event.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaEventRecordWithFlags ( cudaEvent_t event, cudaStream_t stream = 0, unsigned int  flags = 0 )</code><br>
Records an event.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaEventSynchronize ( cudaEvent_t event )</code><br>
Waits for an event to complete.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
</table>

