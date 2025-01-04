<h2>PhOS Support: CUDA Runtime API - EGL Interoperability (0/9)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaEGLStreamConsumerAcquireFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t* pCudaResource, cudaStream_t* pStream, unsigned int timeout)</code><br>
Acquire an image frame from the EGLStream with CUDA as a consumer.
</td>
</tr>
<tr>
<td>470</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaEGLStreamConsumerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream)</code><br>
Connect CUDA to EGLStream as a consumer.
</td>
</tr>
<tr>
<td>471</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaEGLStreamConsumerConnectWithFlags(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, unsigned int flags)</code><br>
Connect CUDA to EGLStream as a consumer with given flags.
</td>
</tr>
<tr>
<td>472</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaEGLStreamConsumerDisconnect(cudaEglStreamConnection* conn)</code><br>
Disconnect CUDA as a consumer to EGLStream.
</td>
</tr>
<tr>
<td>473</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaEGLStreamConsumerReleaseFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t pCudaResource, cudaStream_t* pStream)</code><br>
Releases the last frame acquired from the EGLStream.
</td>
</tr>
<tr>
<td>474</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaEGLStreamProducerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, EGLint width, EGLint height)</code><br>
Connect CUDA to EGLStream as a producer.
</td>
</tr>
<tr>
<td>475</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaEGLStreamProducerDisconnect(cudaEglStreamConnection* conn)</code><br>
Disconnect CUDA as a producer to EGLStream.
</td>
</tr>
<tr>
<td>476</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaEGLStreamProducerPresentFrame(cudaEglStreamConnection* conn, cudaEglFrame eglframe, cudaStream_t* pStream)</code><br>
Present a CUDA eglFrame to the EGLStream with CUDA as a producer.
</td>
</tr>
<tr>
<td>477</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaEGLStreamProducerReturnFrame(cudaEglStreamConnection* conn, cudaEglFrame* eglframe, cudaStream_t* pStream)</code><br>
Return the CUDA eglFrame to the EGLStream last released by the consumer.
</td>
</tr>
<tr>
<td>478</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaEventCreateFromEGLSync(cudaEvent_t* phEvent, EGLSyncKHR eglSync, unsigned int flags)</code><br>
Creates an event from EGLSync object.
</td>
</tr>
<tr>
<td>479</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsEGLRegisterImage(cudaGraphicsResource** pCudaResource, EGLImageKHR image, unsigned int flags)</code><br>
Registers an EGL image.
</td>
</tr>
<tr>
<td>480</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudaError_t cudaGraphicsResourceGetMappedEglFrame(cudaEglFrame* eglFrame, cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel)</code><br>
Get an eglFrame through which to access a registered EGL graphics resource.
</td>
</tr>
<tr>
<td>481</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
