<h2>PhOS Support: CUDA 11.3 - Driver APIs - Stream Management (0/21)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__STREAM.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamAddCallback ( CUstream hStream, CUstreamCallback callback, void* userData, unsigned int  flags )</code><br>
Add a callback to a compute stream.
</td>
</tr>
<tr>
<td>850</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamAttachMemAsync ( CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int  flags )</code><br>
Attach memory to a stream asynchronously.
</td>
</tr>
<tr>
<td>851</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamBeginCapture ( CUstream hStream, CUstreamCaptureMode mode )</code><br>
Begins graph capture on a stream.
</td>
</tr>
<tr>
<td>852</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamCopyAttributes ( CUstream dst, CUstream src )</code><br>
Copies attributes from source stream to destination stream.
</td>
</tr>
<tr>
<td>853</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamCreate ( CUstream* phStream, unsigned int  Flags )</code><br>
Create a stream.
</td>
</tr>
<tr>
<td>854</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamCreateWithPriority ( CUstream* phStream, unsigned int  flags, int  priority )</code><br>
Create a stream with the given priority.
</td>
</tr>
<tr>
<td>855</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamDestroy ( CUstream hStream )</code><br>
Destroys a stream.
</td>
</tr>
<tr>
<td>856</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamEndCapture ( CUstream hStream, CUgraph* phGraph )</code><br>
Ends capture on a stream, returning the captured graph.
</td>
</tr>
<tr>
<td>857</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamGetAttribute ( CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out )</code><br>
Queries stream attribute.
</td>
</tr>
<tr>
<td>858</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamGetCaptureInfo ( CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out )</code><br>
Query capture status of a stream.
</td>
</tr>
<tr>
<td>859</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamGetCaptureInfo_v2 ( CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, size_t* numDependencies_out )</code><br>
Query a stream's capture state (11.3+).
</td>
</tr>
<tr>
<td>860</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamGetCtx ( CUstream hStream, CUcontext* pctx )</code><br>
Query the context associated with a stream.
</td>
</tr>
<tr>
<td>861</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamGetFlags ( CUstream hStream, unsigned int* flags )</code><br>
Query the flags of a given stream.
</td>
</tr>
<tr>
<td>862</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamGetPriority ( CUstream hStream, int* priority )</code><br>
Query the priority of a given stream.
</td>
</tr>
<tr>
<td>863</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamIsCapturing ( CUstream hStream, CUstreamCaptureStatus* captureStatus )</code><br>
Returns a stream's capture status.
</td>
</tr>
<tr>
<td>864</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamQuery ( CUstream hStream )</code><br>
Determine status of a compute stream.
</td>
</tr>
<tr>
<td>865</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamSetAttribute ( CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value )</code><br>
Sets stream attribute.
</td>
</tr>
<tr>
<td>866</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamSynchronize ( CUstream hStream )</code><br>
Wait until a stream's tasks are completed.
</td>
</tr>
<tr>
<td>867</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamUpdateCaptureDependencies ( CUstream hStream, CUgraphNode* dependencies, size_t numDependencies, unsigned int  flags )</code><br>
Update the set of dependencies in a capturing stream (11.3+).
</td>
</tr>
<tr>
<td>868</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuStreamWaitEvent ( CUstream hStream, CUevent hEvent, unsigned int  Flags )</code><br>
Make a compute stream wait on an event.
</td>
</tr>
<tr>
<td>869</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuThreadExchangeStreamCaptureMode ( CUstreamCaptureMode* mode )</code><br>
Swaps the stream capture interaction mode for a thread.
</td>
</tr>
<tr>
<td>870</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
