<h2>PhOS Support: cuDNN 8.0 - Operations Inference (0/103)</h2>

<p>
Documentation: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/api/index.html#cudnn-ops-infer-so-api

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnActivationForward ( cudnnHandle_t handle, const cudnnActivationDescriptor_t activationDesc, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y )</code><br>
Performs forward activation.
</td>
</tr>
<tr>
<td>1800</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnAddTensor ( cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta, const cudnnTensorDescriptor_t cDesc, void *C )</code><br>
Adds two tensors.
</td>
</tr>
<tr>
<td>1801</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBatchNormalizationForwardInference ( cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon )</code><br>
Performs batch normalization forward inference.
</td>
</tr>
<tr>
<td>1802</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCopyAlgorithmDescriptor ( cudnnAlgorithmDescriptor_t dest, const cudnnAlgorithmDescriptor_t src )</code><br>
Copies an algorithm descriptor.
</td>
</tr>
<tr>
<td>1803</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreate ( cudnnHandle_t *handle )</code><br>
Creates a cuDNN handle.
</td>
</tr>
<tr>
<td>1804</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateActivationDescriptor ( cudnnActivationDescriptor_t *activationDesc )</code><br>
Creates an activation descriptor object.
</td>
</tr>
<tr>
<td>1805</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateAlgorithmDescriptor ( cudnnAlgorithmDescriptor_t *algoDesc )</code><br>
Creates an algorithm descriptor object. (Deprecated in cuDNN 8.0)
</td>
</tr>
<tr>
<td>1806</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateAlgorithmPerformance ( cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate )</code><br>
Creates multiple algorithm performance objects.
</td>
</tr>
<tr>
<td>1807</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateDropoutDescriptor ( cudnnDropoutDescriptor_t *dropoutDesc )</code><br>
Creates a dropout descriptor object.
</td>
</tr>
<tr>
<td>1808</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateFilterDescriptor ( cudnnFilterDescriptor_t *filterDesc )</code><br>
Creates a filter descriptor object.
</td>
</tr>
<tr>
<td>1809</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateLRNDescriptor ( cudnnLRNDescriptor_t *normDesc )</code><br>
Creates a local response normalization descriptor object.
</td>
</tr>
<tr>
<td>1810</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateOpTensorDescriptor ( cudnnOpTensorDescriptor_t *opTensorDesc )</code><br>
Creates an operation tensor descriptor object.
</td>
</tr>
<tr>
<td>1811</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreatePoolingDescriptor ( cudnnPoolingDescriptor_t *poolingDesc )</code><br>
Creates a pooling descriptor object.
</td>
</tr>
<tr>
<td>1812</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateReduceTensorDescriptor ( cudnnReduceTensorDescriptor_t *reduceTensorDesc )</code><br>
Creates a reduce tensor descriptor object.
</td>
</tr>
<tr>
<td>1813</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateSpatialTransformerDescriptor ( cudnnSpatialTransformerDescriptor_t *stDesc )</code><br>
Creates a spatial transformer descriptor object.
</td>
</tr>
<tr>
<td>1814</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateTensorDescriptor ( cudnnTensorDescriptor_t *tensorDesc )</code><br>
Creates a tensor descriptor object.
</td>
</tr>
<tr>
<td>1815</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateTensorTransformDescriptor ( cudnnTensorTransformDescriptor_t *transformDesc )</code><br>
Creates a tensor transform descriptor object.
</td>
</tr>
<tr>
<td>1816</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDeriveBNTensorDescriptor ( cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc, cudnnBatchNormMode_t mode )</code><br>
Derives a batch normalization tensor descriptor.
</td>
</tr>
<tr>
<td>1817</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDeriveNormTensorDescriptor ( cudnnTensorDescriptor_t derivedNormDesc, const cudnnTensorDescriptor_t xDesc, cudnnNormMode_t mode )</code><br>
Derives a normalization tensor descriptor.
</td>
</tr>
<tr>
<td>1818</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroy ( cudnnHandle_t handle )</code><br>
Releases resources used by the cuDNN handle.
</td>
</tr>
<tr>
<td>1819</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyActivationDescriptor ( cudnnActivationDescriptor_t activationDesc )</code><br>
Destroys an activation descriptor object.
</td>
</tr>
<tr>
<td>1820</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyAlgorithmDescriptor ( cudnnAlgorithmDescriptor_t algoDesc )</code><br>
Destroys an algorithm descriptor object.
</td>
</tr>
<tr>
<td>1821</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyAlgorithmPerformance ( cudnnAlgorithmPerformance_t algoPerf )</code><br>
Destroys an algorithm performance object.
</td>
</tr>
<tr>
<td>1822</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyDropoutDescriptor ( cudnnDropoutDescriptor_t dropoutDesc )</code><br>
Destroys a dropout descriptor object.
</td>
</tr>
<tr>
<td>1823</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyFilterDescriptor ( cudnnFilterDescriptor_t filterDesc )</code><br>
Destroys a filter descriptor object.
</td>
</tr>
<tr>
<td>1824</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyLRNDescriptor ( cudnnLRNDescriptor_t normDesc )</code><br>
Destroys a local response normalization descriptor object.
</td>
</tr>
<tr>
<td>1825</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyOpTensorDescriptor ( cudnnOpTensorDescriptor_t opTensorDesc )</code><br>
Destroys an operation tensor descriptor object.
</td>
</tr>
<tr>
<td>1826</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyPoolingDescriptor ( cudnnPoolingDescriptor_t poolingDesc )</code><br>
Destroys a pooling descriptor object.
</td>
</tr>
<tr>
<td>1827</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyReduceTensorDescriptor ( cudnnReduceTensorDescriptor_t reduceTensorDesc )</code><br>
Destroys a reduce tensor descriptor object.
</td>
</tr>
<tr>
<td>1828</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroySpatialTransformerDescriptor ( cudnnSpatialTransformerDescriptor_t stDesc )</code><br>
Destroys a spatial transformer descriptor object.
</td>
</tr>
<tr>
<td>1829</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyTensorDescriptor ( cudnnTensorDescriptor_t tensorDesc )</code><br>
Destroys a tensor descriptor object.
</td>
</tr>
<tr>
<td>1830</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyTensorTransformDescriptor ( cudnnTensorTransformDescriptor_t transformDesc )</code><br>
Destroys a tensor transform descriptor object.
</td>
</tr>
<tr>
<td>1831</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDivisiveNormalizationForward ( cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *means, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y, void *temp, size_t tempSizeInBytes )</code><br>
Performs divisive normalization forward.
</td>
</tr>
<tr>
<td>1832</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDropoutForward ( cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, void *reserveSpace, size_t reserveSpaceSizeInBytes )</code><br>
Performs dropout forward.
</td>
</tr>
<tr>
<td>1833</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDropoutGetReserveSpaceSize ( cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes )</code><br>
Gets the size of the reserve space for dropout.
</td>
</tr>
<tr>
<td>1834</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDropoutGetStatesSize ( cudnnHandle_t handle, size_t *sizeInBytes )</code><br>
Gets the size of the dropout states.
</td>
</tr>
<tr>
<td>1835</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetActivationDescriptor ( cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t *mode, cudnnNanPropagation_t *reluNanOpt, double *coef )</code><br>
Gets the activation descriptor.
</td>
</tr>
<tr>
<td>1836</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetActivationDescriptorSwishBeta ( cudnnActivationDescriptor_t activationDesc, double *swishBeta )</code><br>
Gets the Swish beta value from the activation descriptor.
</td>
</tr>
<tr>
<td>1837</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetAlgorithmDescriptor ( cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algo )</code><br>
Gets the algorithm descriptor.
</td>
</tr>
<tr>
<td>1838</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetAlgorithmPerformance ( cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithm_t *algo, cudnnStatus_t *status, float *time, size_t *memory )</code><br>
Gets the algorithm performance.
</td>
</tr>
<tr>
<td>1839</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetAlgorithmSpaceSize ( cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, size_t *algoSpaceSizeInBytes )</code><br>
Gets the size of the algorithm space.
</td>
</tr>
<tr>
<td>1840</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetCallback ( cudnnHandle_t handle, cudnnCallback_t *callback )</code><br>
Gets the callback function.
</td>
</tr>
<tr>
<td>1841</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetCudartVersion ( int *version )</code><br>
Gets the CUDART version.
</td>
</tr>
<tr>
<td>1842</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetDropoutDescriptor ( cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float *dropout, void **states, unsigned long long *seed )</code><br>
Gets the dropout descriptor.
</td>
</tr>
<tr>
<td>1843</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetErrorString ( cudnnStatus_t status )</code><br>
Returns a string describing the error code.
</td>
</tr>
<tr>
<td>1844</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetFilter4dDescriptor ( cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType, cudnnTensorFormat_t *format, int *k, int *c, int *h, int *w )</code><br>
Gets the 4D filter descriptor.
</td>
</tr>
<tr>
<td>1845</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetFilterNdDescriptor ( cudnnFilterDescriptor_t filterDesc, int nbDimsRequested, cudnnDataType_t *dataType, cudnnTensorFormat_t *format, int *nbDims, int *filterDimA )</code><br>
Gets the ND filter descriptor.
</td>
</tr>
<tr>
<td>1846</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetFilterSizeInBytes ( cudnnFilterDescriptor_t filterDesc, size_t *size )</code><br>
Gets the size of the filter in bytes.
</td>
</tr>
<tr>
<td>1847</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetLRNDescriptor ( cudnnLRNDescriptor_t normDesc, unsigned *lrnN, double *lrnAlpha, double *lrnBeta, double *lrnK )</code><br>
Gets the LRN descriptor.
</td>
</tr>
<tr>
<td>1848</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetOpTensorDescriptor ( cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t *opTensorOp, cudnnDataType_t *opTensorCompType, cudnnNanPropagation_t *opTensorNanOpt )</code><br>
Gets the operation tensor descriptor.
</td>
</tr>
<tr>
<td>1849</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetPooling2dDescriptor ( cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight, int *windowWidth, int *verticalPadding, int *horizontalPadding, int *verticalStride, int *horizontalStride )</code><br>
Gets the 2D pooling descriptor.
</td>
</tr>
<tr>
<td>1850</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetPooling2dForwardOutputDim ( const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc, int *n, int *c, int *h, int *w )</code><br>
Gets the output dimensions of the 2D pooling forward operation.
</td>
</tr>
<tr>
<td>1851</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetPoolingNdDescriptor ( cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested, cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt, int *nbDims, int *windowDimA, int *paddingA, int *strideA )</code><br>
Gets the ND pooling descriptor.
</td>
</tr>
<tr>
<td>1852</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetPoolingNdForwardOutputDim ( const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc, int nbDims, int *outputTensorDimA )</code><br>
Gets the output dimensions of the ND pooling forward operation.
</td>
</tr>
<tr>
<td>1853</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetProperty ( cudnnPropertyType_t type, int *value )</code><br>
Gets a property of cuDNN.
</td>
</tr>
<tr>
<td>1854</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetReduceTensorDescriptor ( cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t *reduceTensorOp, cudnnDataType_t *reduceTensorCompType, cudnnNanPropagation_t *reduceTensorNanOpt, cudnnReduceTensorIndices_t *reduceTensorIndices, cudnnIndicesType_t *reduceTensorIndicesType )</code><br>
Gets the reduce tensor descriptor.
</td>
</tr>
<tr>
<td>1855</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetReductionIndicesSize ( cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes )</code><br>
Gets the size of the reduction indices.
</td>
</tr>
<tr>
<td>1856</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetReductionWorkspaceSize ( cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes )</code><br>
Gets the size of the reduction workspace.
</td>
</tr>
<tr>
<td>1857</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetStream ( cudnnHandle_t handle, cudaStream_t *streamId )</code><br>
Gets the CUDA stream associated with the cuDNN handle.
</td>
</tr>
<tr>
<td>1858</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetTensor4dDescriptor ( cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t *dataType, int *n, int *c, int *h, int *w, int *nStride, int *cStride, int *hStride, int *wStride )</code><br>
Gets the 4D tensor descriptor.
</td>
</tr>
<tr>
<td>1859</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetTensorNdDescriptor ( cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested, cudnnDataType_t *dataType, int *nbDims, int *dimA, int *strideA )</code><br>
Gets the ND tensor descriptor.
</td>
</tr>
<tr>
<td>1860</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetTensorSizeInBytes ( cudnnTensorDescriptor_t tensorDesc, size_t *size )</code><br>
Gets the size of the tensor in bytes.
</td>
</tr>
<tr>
<td>1861</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetTensorTransformDescriptor ( cudnnTensorTransformDescriptor_t transformDesc, cudnnTensorFormat_t *srcFormat, cudnnTensorFormat_t *destFormat, cudnnTensorTransformMode_t *mode, cudnnTensorTransformOp_t *op, cudnnDataType_t *dataType, cudnnNanPropagation_t *nanOpt )</code><br>
Gets the tensor transform descriptor.
</td>
</tr>
<tr>
<td>1862</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetVersion ( void )</code><br>
Gets the version of cuDNN.
</td>
</tr>
<tr>
<td>1863</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnInitTransformDest ( cudnnTensorTransformDescriptor_t transformDesc, cudnnTensorDescriptor_t destDesc, const cudnnTensorDescriptor_t srcDesc )</code><br>
Initializes the destination tensor descriptor for a transform.
</td>
</tr>
<tr>
<td>1864</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnLRNCrossChannelForward ( cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t mode, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y )</code><br>
Performs LRN cross-channel forward.
</td>
</tr>
<tr>
<td>1865</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnNormalizationForwardInference ( cudnnHandle_t handle, cudnnNormMode_t mode, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t normScaleBiasMeanVarDesc, const void *normScale, const void *normBias, const void *estimatedMean, const void *estimatedVariance, double epsilon )</code><br>
Performs normalization forward inference.
</td>
</tr>
<tr>
<td>1866</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnOpsInferVersionCheck ( cudnnHandle_t handle, int *version )</code><br>
Checks the version of cuDNN operations inference.
</td>
</tr>
<tr>
<td>1867</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnOpTensor ( cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc, const void *alpha1, const cudnnTensorDescriptor_t aDesc, const void *A, const void *alpha2, const cudnnTensorDescriptor_t bDesc, const void *B, const void *beta, const cudnnTensorDescriptor_t cDesc, void *C )</code><br>
Performs a tensor operation.
</td>
</tr>
<tr>
<td>1868</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnPoolingForward ( cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y )</code><br>
Performs pooling forward.
</td>
</tr>
<tr>
<td>1869</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnQueryRuntimeError ( cudnnHandle_t handle, cudnnStatus_t *rstatus, cudnnErrQueryMode_t mode, cudnnRuntimeError_t *error )</code><br>
Queries runtime error.
</td>
</tr>
<tr>
<td>1870</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnReduceTensor ( cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, void *indices, size_t indicesSizeInBytes, void *workspace, size_t workspaceSizeInBytes, const void *alpha, const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta, const cudnnTensorDescriptor_t cDesc, void *C )</code><br>
Performs tensor reduction.
</td>
</tr>
<tr>
<td>1871</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRestoreAlgorithm ( cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, const void *algoData, size_t algoDataSizeInBytes )</code><br>
Restores an algorithm.
</td>
</tr>
<tr>
<td>1872</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRestoreDropoutDescriptor ( cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout, void *states, size_t stateSizeInBytes, unsigned long long seed )</code><br>
Restores a dropout descriptor.
</td>
</tr>
<tr>
<td>1873</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSaveAlgorithm ( cudnnHandle_t handle, const cudnnAlgorithmDescriptor_t algoDesc, void *algoData, size_t algoDataSizeInBytes )</code><br>
Saves an algorithm.
</td>
</tr>
<tr>
<td>1874</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnScaleTensor ( cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *alpha )</code><br>
Scales a tensor.
</td>
</tr>
<tr>
<td>1875</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetActivationDescriptor ( cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt, double coef )</code><br>
Sets the activation descriptor.
</td>
</tr>
<tr>
<td>1876</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetActivationDescriptorSwishBeta ( cudnnActivationDescriptor_t activationDesc, double swishBeta )</code><br>
Sets the Swish beta value in the activation descriptor.
</td>
</tr>
<tr>
<td>1877</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetAlgorithmDescriptor ( cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algo )</code><br>
Sets the algorithm descriptor.
</td>
</tr>
<tr>
<td>1878</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetAlgorithmPerformance ( cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithm_t algo, cudnnStatus_t status, float time, size_t memory )</code><br>
Sets the algorithm performance.
</td>
</tr>
<tr>
<td>1879</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetCallback ( cudnnHandle_t handle, cudnnCallback_t callback )</code><br>
Sets a callback function.
</td>
</tr>
<tr>
<td>1880</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetDropoutDescriptor ( cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout, void *states, size_t stateSizeInBytes, unsigned long long seed )</code><br>
Sets the dropout descriptor.
</td>
</tr>
<tr>
<td>1881</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetFilter4dDescriptor ( cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int k, int c, int h, int w )</code><br>
Sets the 4D filter descriptor.
</td>
</tr>
<tr>
<td>1882</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetFilterNdDescriptor ( cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int nbDims, const int *filterDimA )</code><br>
Sets the ND filter descriptor.
</td>
</tr>
<tr>
<td>1883</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetLRNDescriptor ( cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK )</code><br>
Sets the LRN descriptor.
</td>
</tr>
<tr>
<td>1884</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetOpTensorDescriptor ( cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp, cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt )</code><br>
Sets the operation tensor descriptor.
</td>
</tr>
<tr>
<td>1885</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetPooling2dDescriptor ( cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode, cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride )</code><br>
Sets the 2D pooling descriptor.
</td>
</tr>
<tr>
<td>1886</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetPoolingNdDescriptor ( cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode, cudnnNanPropagation_t maxpoolingNanOpt, int nbDims, const int *windowDimA, const int *paddingA, const int *strideA )</code><br>
Sets the ND pooling descriptor.
</td>
</tr>
<tr>
<td>1887</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetReduceTensorDescriptor ( cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t reduceTensorOp, cudnnDataType_t reduceTensorCompType, cudnnNanPropagation_t reduceTensorNanOpt, cudnnReduceTensorIndices_t reduceTensorIndices, cudnnIndicesType_t reduceTensorIndicesType )</code><br>
Sets the reduce tensor descriptor.
</td>
</tr>
<tr>
<td>1888</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor ( cudnnSpatialTransformerDescriptor_t stDesc, cudnnSpatialTransformerMode_t mode, cudnnDataType_t dataType, int nbDims, const int *dimA )</code><br>
Sets the spatial transformer ND descriptor.
</td>
</tr>
<tr>
<td>1889</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetStream ( cudnnHandle_t handle, cudaStream_t streamId )</code><br>
Sets the CUDA stream associated with the cuDNN handle.
</td>
</tr>
<tr>
<td>1890</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetTensor ( cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *valuePtr )</code><br>
Sets the tensor to a specific value.
</td>
</tr>
<tr>
<td>1891</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetTensor4dDescriptor ( cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w )</code><br>
Sets the 4D tensor descriptor.
</td>
</tr>
<tr>
<td>1892</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetTensor4dDescriptorEx ( cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int n, int c, int h, int w, int nStride, int cStride, int hStride, int wStride )</code><br>
Sets the 4D tensor descriptor with strides.
</td>
</tr>
<tr>
<td>1893</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetTensorNdDescriptor ( cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims, const int *dimA, const int *strideA )</code><br>
Sets the ND tensor descriptor.
</td>
</tr>
<tr>
<td>1894</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetTensorNdDescriptorEx ( cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int nbDims, const int *dimA )</code><br>
Sets the ND tensor descriptor with format.
</td>
</tr>
<tr>
<td>1895</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetTensorTransformDescriptor ( cudnnTensorTransformDescriptor_t transformDesc, cudnnTensorFormat_t srcFormat, cudnnTensorFormat_t destFormat, cudnnTensorTransformMode_t mode, cudnnTensorTransformOp_t op, cudnnDataType_t dataType, cudnnNanPropagation_t nanOpt )</code><br>
Sets the tensor transform descriptor.
</td>
</tr>
<tr>
<td>1896</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSoftmaxForward ( cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y )</code><br>
Performs softmax forward.
</td>
</tr>
<tr>
<td>1897</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSpatialTfGridGeneratorForward ( cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc, const void *theta, void *grid )</code><br>
Generates a grid for spatial transformer forward.
</td>
</tr>
<tr>
<td>1898</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSpatialTfSamplerForward ( cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y, const void *grid )</code><br>
Performs spatial transformer sampling forward.
</td>
</tr>
<tr>
<td>1899</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnTransformFilter ( cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transformDesc, const void *alpha, const cudnnFilterDescriptor_t srcDesc, const void *srcData, const void *beta, const cudnnFilterDescriptor_t destDesc, void *destData )</code><br>
Transforms a filter.
</td>
</tr>
<tr>
<td>1900</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnTransformTensor ( cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y )</code><br>
Transforms a tensor.
</td>
</tr>
<tr>
<td>1901</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnTransformTensorEx ( cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transformDesc, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y )</code><br>
Transforms a tensor with extended options.
</td>
</tr>
<tr>
<td>1902</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
