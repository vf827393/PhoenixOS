<h2>PhOS Support: CUDA 11.3 - Runtime APIs - Peer Device Memory Access (0/3)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceCanAccessPeer ( int* canAccessPeer, int  device, int  peerDevice )</code><br>
Queries if a device may directly access a peer device's memory.
</td>
</tr>
<tr>
<td>410</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceDisablePeerAccess ( int  peerDevice )</code><br>
Disables direct access to memory allocations on a peer device.
</td>
</tr>
<tr>
<td>411</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaDeviceEnablePeerAccess ( int  peerDevice, unsigned int  flags )</code><br>
Enables direct access to memory allocations on a peer device.
</td>
</tr>
<tr>
<td>412</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
