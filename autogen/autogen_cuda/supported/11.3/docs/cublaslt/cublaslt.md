<h2>PhOS Support: CUDA 11.3 - cuBLASLt API Functions (0/40)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cublas/index.html#cublasLt-api-ref

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtCreate(cublasLtHandle_t *lighthandle)</code><br>
Initializes the cuBLASLt library and creates a handle to an opaque structure holding the cuBLASLt library context.
</td>
</tr>
<tr>
<td>1500</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle)</code><br>
Releases hardware resources used by the cuBLASLt library.
</td>
</tr>
<tr>
<td>1501</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>size_t cublasLtGetCudartVersion(void)</code><br>
Returns the version number of the CUDA Runtime library.
</td>
</tr>
<tr>
<td>1502</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtGetProperty(libraryPropertyType type, int *value)</code><br>
Returns the value of the requested property.
</td>
</tr>
<tr>
<td>1503</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>size_t cublasLtGetVersion(void)</code><br>
Returns the version number of cuBLASLt library.
</td>
</tr>
<tr>
<td>1504</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback)</code><br>
Sets the logging callback function.
</td>
</tr>
<tr>
<td>1505</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtLoggerSetFile(FILE* file)</code><br>
Sets the logging output file.
</td>
</tr>
<tr>
<td>1506</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtLoggerOpenFile(const char* logFile)</code><br>
Opens a logging output file in the given path.
</td>
</tr>
<tr>
<td>1507</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtLoggerSetLevel(int level)</code><br>
Sets the value of the logging level.
</td>
</tr>
<tr>
<td>1508</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtLoggerSetMask(int mask)</code><br>
Sets the value of the logging mask.
</td>
</tr>
<tr>
<td>1509</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtLoggerForceDisable(void)</code><br>
Disables logging for the entire run.
</td>
</tr>
<tr>
<td>1510</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void *alpha, const void *A, cublasLtMatrixLayout_t Adesc, const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta, const void *C, cublasLtMatrixLayout_t Cdesc, void *D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo, void *workspace, size_t workspaceSizeInBytes, cudaStream_t stream)</code><br>
Computes the matrix multiplication of matrices A and B to produce the output matrix D.
</td>
</tr>
<tr>
<td>1511</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t *algo, cublasLtMatmulAlgoCapAttributes_t attr, void *buf, size_t sizeInBytes, size_t *sizeWritten)</code><br>
Returns the value of the queried capability attribute for an initialized cublasLtMatmulAlgo_t descriptor structure.
</td>
</tr>
<tr>
<td>1512</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulAlgoCheck(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo, cublasLtMatmulHeuristicResult_t *result)</code><br>
Performs the correctness check on the matrix multiply algorithm descriptor for the matrix multiply operation.
</td>
</tr>
<tr>
<td>1513</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t *algo, cublasLtMatmulAlgoConfigAttributes_t attr, void *buf, size_t sizeInBytes, size_t *sizeWritten)</code><br>
Returns the value of the queried configuration attribute for an initialized cublasLtMatmulAlgo_t descriptor.
</td>
</tr>
<tr>
<td>1514</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t *algo, cublasLtMatmulAlgoConfigAttributes_t attr, const void *buf, size_t sizeInBytes)</code><br>
Sets the value of the specified configuration attribute for an initialized cublasLtMatmulAlgo_t descriptor.
</td>
</tr>
<tr>
<td>1515</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference, int requestedAlgoCount, cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int *returnAlgoCount)</code><br>
Retrieves the possible algorithms for the matrix multiply operation.
</td>
</tr>
<tr>
<td>1516</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulAlgoGetIds(cublasLtHandle_t lightHandle, cublasComputeType_t computeType, cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t Dtype, int requestedAlgoCount, int algoIdsArray[], int *returnAlgoCount)</code><br>
Retrieves the IDs of all the matrix multiply algorithms that are valid.
</td>
</tr>
<tr>
<td>1517</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle, cublasComputeType_t computeType, cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t Dtype, int algoId, cublasLtMatmulAlgo_t *algo)</code><br>
Initializes the matrix multiply algorithm structure for the cublasLtMatmul().
</td>
</tr>
<tr>
<td>1518</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType)</code><br>
Creates a matrix multiply descriptor.
</td>
</tr>
<tr>
<td>1519</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulDescInit(cublasLtMatmulDesc_t matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType)</code><br>
Initializes a matrix multiply descriptor in a previously allocated one.
</td>
</tr>
<tr>
<td>1520</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc)</code><br>
Destroys a previously created matrix multiply descriptor object.
</td>
</tr>
<tr>
<td>1521</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, void *buf, size_t sizeInBytes, size_t *sizeWritten)</code><br>
Returns the value of the queried attribute belonging to a previously created matrix multiply descriptor.
</td>
</tr>
<tr>
<td>1522</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, const void *buf, size_t sizeInBytes)</code><br>
Sets the value of the specified attribute belonging to a previously created matrix multiply descriptor.
</td>
</tr>
<tr>
<td>1523</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref)</code><br>
Creates a matrix multiply heuristic search preferences descriptor.
</td>
</tr>
<tr>
<td>1524</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulPreferenceInit(cublasLtMatmulPreference_t pref)</code><br>
Initializes a matrix multiply heuristic search preferences descriptor in a previously allocated one.
</td>
</tr>
<tr>
<td>1525</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref)</code><br>
Destroys a previously created matrix multiply preferences descriptor object.
</td>
</tr>
<tr>
<td>1526</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, void *buf, size_t sizeInBytes, size_t *sizeWritten)</code><br>
Returns the value of the queried attribute belonging to a previously created matrix multiply heuristic search preferences descriptor.
</td>
</tr>
<tr>
<td>1527</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const void *buf, size_t sizeInBytes)</code><br>
Sets the value of the specified attribute belonging to a previously created matrix multiply preferences descriptor.
</td>
</tr>
<tr>
<td>1528</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld)</code><br>
Creates a matrix layout descriptor.
</td>
</tr>
<tr>
<td>1529</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatrixLayoutInit(cublasLtMatrixLayout_t matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld)</code><br>
Initializes a matrix layout descriptor in a previously allocated one.
</td>
</tr>
<tr>
<td>1530</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout)</code><br>
Destroys a previously created matrix layout descriptor object.
</td>
</tr>
<tr>
<td>1531</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, void *buf, size_t sizeInBytes, size_t *sizeWritten)</code><br>
Returns the value of the queried attribute belonging to the specified matrix layout descriptor.
</td>
</tr>
<tr>
<td>1532</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, const void *buf, size_t sizeInBytes)</code><br>
Sets the value of the specified attribute belonging to a previously created matrix layout descriptor.
</td>
</tr>
<tr>
<td>1533</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatrixTransform(cublasLtHandle_t lightHandle, cublasLtMatrixTransformDesc_t transformDesc, const void *alpha, const void *A, cublasLtMatrixLayout_t Adesc, const void *beta, const void *B, cublasLtMatrixLayout_t Bdesc, void *C, cublasLtMatrixLayout_t Cdesc, cudaStream_t stream)</code><br>
Computes the matrix transformation operation on the input matrices A and B, to produce the output matrix C.
</td>
</tr>
<tr>
<td>1534</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t *transformDesc, cudaDataType scaleType)</code><br>
Creates a matrix transform descriptor.
</td>
</tr>
<tr>
<td>1535</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatrixTransformDescInit(cublasLtMatrixTransformDesc_t transformDesc, cudaDataType scaleType)</code><br>
Initializes a matrix transform descriptor in a previously allocated one.
</td>
</tr>
<tr>
<td>1536</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t transformDesc)</code><br>
Destroys a previously created matrix transform descriptor object.
</td>
</tr>
<tr>
<td>1537</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatrixTransformDescGetAttribute(cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, void *buf, size_t sizeInBytes, size_t *sizeWritten)</code><br>
Returns the value of the queried attribute belonging to a previously created matrix transform descriptor.
</td>
</tr>
<tr>
<td>1538</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cublasStatus_t cublasLtMatrixTransformDescSetAttribute(cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, const void *buf, size_t sizeInBytes)</code><br>
Sets the value of the specified attribute belonging to a previously created matrix transform descriptor.
</td>
</tr>
<tr>
<td>1539</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
