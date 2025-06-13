<h2>PhOS Support: CUDA 11.3 - cuBLASXt Helper Functions (0/9)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#cublasxt-api-ref

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtCreate(cublasXtHandle_t *handle)</code><br>
Initializes the cuBLASXt API and creates a handle to an opaque structure holding the cuBLASXt API context.
</td>
</tr>
<tr>
<td>1600</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtDestroy(cublasXtHandle_t handle)</code><br>
Releases hardware resources used by the cuBLASXt API context.
</td>
</tr>
<tr>
<td>1601</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtDeviceSelect(cublasXtHandle_t handle, int nbDevices, int deviceId[])</code><br>
Allows the user to provide the number of GPU devices and their respective Ids that will participate in the subsequent cuBLASXt API Math function calls.
</td>
</tr>
<tr>
<td>1602</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtSetBlockDim(cublasXtHandle_t handle, int blockDim)</code><br>
Sets the block dimension used for the tiling of the matrices for the subsequent Math function calls.
</td>
</tr>
<tr>
<td>1603</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtGetBlockDim(cublasXtHandle_t handle, int *blockDim)</code><br>
Queries the block dimension used for the tiling of the matrices.
</td>
</tr>
<tr>
<td>1604</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtSetCpuRoutine(cublasXtHandle_t handle, cublasXtBlasOp_t blasOp, cublasXtOpType_t type, void *blasFunctor)</code><br>
Provides a CPU implementation of the corresponding BLAS routine.
</td>
</tr>
<tr>
<td>1605</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtSetCpuRatio(cublasXtHandle_t handle, cublasXtBlasOp_t blasOp, cublasXtOpType_t type, float ratio)</code><br>
Defines the percentage of workload that should be done on a CPU in the context of a hybrid computation.
</td>
</tr>
<tr>
<td>1606</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtSetPinningMemMode(cublasXtHandle_t handle, cublasXtPinningMemMode_t mode)</code><br>
Enables or disables the Pinning Memory mode.
</td>
</tr>
<tr>
<td>1607</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasXtGetPinningMemMode(cublasXtHandle_t handle, cublasXtPinningMemMode_t *mode)</code><br>
Queries the Pinning Memory mode.
</td>
</tr>
<tr>
<td>1608</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
