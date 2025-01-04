<h2>PhOS Support: CUDA 11.3 - Driver APIs - Primary Context Management (0/5)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDevicePrimaryCtxGetState ( CUdevice dev, unsigned int* flags, int* active )</code><br>
Get the state of the primary context.
</td>
</tr>
<tr>
<td>670</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDevicePrimaryCtxRelease ( CUdevice dev )</code><br>
Release the primary context on the GPU.
</td>
</tr>
<tr>
<td>671</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDevicePrimaryCtxReset ( CUdevice dev )</code><br>
Destroy all allocations and reset all state on the primary context.
</td>
</tr>
<tr>
<td>672</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDevicePrimaryCtxRetain ( CUcontext* pctx, CUdevice dev )</code><br>
Retain the primary context on the GPU.
</td>
</tr>
<tr>
<td>673</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuDevicePrimaryCtxSetFlags ( CUdevice dev, unsigned int flags )</code><br>
Set flags for the primary context.
</td>
</tr>
<tr>
<td>674</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
