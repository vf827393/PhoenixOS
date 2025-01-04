<h2>PhOS Support: CUDA 11.3 - Driver APIs - Event Management (0/7)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__EVENT.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEventCreate ( CUevent* phEvent, unsigned int  Flags )</code><br>
Creates an event.
</td>
</tr>
<tr>
<td>880</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEventDestroy ( CUevent hEvent )</code><br>
Destroys an event.
</td>
</tr>
<tr>
<td>881</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEventElapsedTime ( float* pMilliseconds, CUevent hStart, CUevent hEnd )</code><br>
Computes the elapsed time between two events.
</td>
</tr>
<tr>
<td>882</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEventQuery ( CUevent hEvent )</code><br>
Queries an event's status.
</td>
</tr>
<tr>
<td>883</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEventRecord ( CUevent hEvent, CUstream hStream )</code><br>
Records an event.
</td>
</tr>
<tr>
<td>884</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEventRecordWithFlags ( CUevent hEvent, CUstream hStream, unsigned int  flags )</code><br>
Records an event.
</td>
</tr>
<tr>
<td>885</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuEventSynchronize ( CUevent hEvent )</code><br>
Waits for an event to complete.
</td>
</tr>
<tr>
<td>886</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
