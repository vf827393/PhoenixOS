# [Stream Management (0/22)](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__STREAM.html)

<table>
<tr>
<th>Supported</th>
<th>Test Passed</th>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaCtxResetPersistingL2Cache ( void )</code><br>
Resets all persisting lines in cache to normal status.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamAddCallback ( cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int  flags )</code><br>
Add a callback to a compute stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamAttachMemAsync ( cudaStream_t stream, void* devPtr, size_t length = 0, unsigned int  flags = cudaMemAttachSingle )</code><br>
Attach memory to a stream asynchronously.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamBeginCapture ( cudaStream_t stream, cudaStreamCaptureMode mode )</code><br>
Begins graph capture on a stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamCopyAttributes ( cudaStream_t dst, cudaStream_t src )</code><br>
Copies attributes from source stream to destination stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamCreate ( cudaStream_t* pStream )</code><br>
Create an asynchronous stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamCreateWithFlags ( cudaStream_t* pStream, unsigned int  flags )</code><br>
Create an asynchronous stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamCreateWithPriority ( cudaStream_t* pStream, unsigned int  flags, int  priority )</code><br>
Create an asynchronous stream with the specified priority.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamDestroy ( cudaStream_t stream )</code><br>
Destroys and cleans up an asynchronous stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamEndCapture ( cudaStream_t stream, cudaGraph_t* pGraph )</code><br>
Ends capture on a stream, returning the captured graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamGetAttribute ( cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out )</code><br>
Queries stream attribute.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamGetCaptureInfo ( cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus, unsigned long long* pId )</code><br>
Query capture status of a stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamGetCaptureInfo_v2 ( cudaStream_t stream, cudaStreamCaptureStatus ** captureStatus_out, unsigned long long* id_out = 0, cudaGraph_t* graph_out = 0, const cudaGraphNode_t** dependencies_out = 0, size_t* numDependencies_out = 0 )</code><br>
Query a stream's capture state (11.3+).
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamGetFlags ( cudaStream_t hStream, unsigned int* flags )</code><br>
Query the flags of a stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamGetPriority ( cudaStream_t hStream, int* priority )</code><br>
Query the priority of a stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamIsCapturing ( cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus )</code><br>
Returns a stream's capture status.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamQuery ( cudaStream_t stream )</code><br>
Queries an asynchronous stream for completion status.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamSetAttribute ( cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value )</code><br>
Sets stream attribute.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamSynchronize ( cudaStream_t stream )</code><br>
Waits for stream tasks to complete.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamUpdateCaptureDependencies ( cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int  flags = 0 )</code><br>
Update the set of dependencies in a capturing stream (11.3+).
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaStreamWaitEvent ( cudaStream_t stream, cudaEvent_t event, unsigned int  flags = 0 )</code><br>
Make a compute stream wait on an event.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaThreadExchangeStreamCaptureMode ( cudaStreamCaptureMode ** mode )</code><br>
Swaps the stream capture interaction mode for a thread.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
</table>
