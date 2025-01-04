<h2>PhOS Support: cuDNN 8.0 - Advanced Inference (0/53)</h2>

<p>
Documentation: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/api/index.html#cudnn-adv-infer-so-api

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnAdvInferVersionCheck(void)</code><br>
Checks the version of the AdvInfer subset of the library.
</td>
</tr>
<tr>
<td>2300</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBuildRNNDynamic(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int32_t miniBatch)</code><br>
Compiles the RNN persistent code using CUDA runtime compilation library (NVRTC).
</td>
</tr>
<tr>
<td>2301</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc)</code><br>
Creates an attention descriptor object.
</td>
</tr>
<tr>
<td>2302</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, const int minibatch, const cudnnDataType_t dataType, cudnnPersistentRNNPlan_t *plan)</code><br>
Creates a plan to execute persistent RNNs.
</td>
</tr>
<tr>
<td>2303</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *RNNDataDesc)</code><br>
Creates a RNN data descriptor object.
</td>
</tr>
<tr>
<td>2304</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc)</code><br>
Creates a generic RNN descriptor object.
</td>
</tr>
<tr>
<td>2305</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc)</code><br>
Creates a sequence data descriptor object.
</td>
</tr>
<tr>
<td>2306</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc)</code><br>
Destroys the attention descriptor object.
</td>
</tr>
<tr>
<td>2307</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan)</code><br>
Destroys a previously created persistent RNN plan object.
</td>
</tr>
<tr>
<td>2308</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t RNNDataDesc)</code><br>
Destroys a previously created RNN data descriptor object.
</td>
</tr>
<tr>
<td>2309</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc)</code><br>
Destroys a previously created RNN descriptor object.
</td>
</tr>
<tr>
<td>2310</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc)</code><br>
Destroys the sequence data descriptor object.
</td>
</tr>
<tr>
<td>2311</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes)</code><br>
Attempts all available cuDNN algorithms for RNN forward inference.
</td>
</tr>
<tr>
<td>2312</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetAttnDescriptor(cudnnAttnDescriptor_t attnDesc, unsigned *attnMode, int *nHeads, double *smScaler, cudnnDataType_t *dataType, cudnnDataType_t *computePrec, cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc, cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize, int *kSize, int *vSize, int *qProjSize, int *kProjSize, int *vProjSize, int *oProjSize, int *qoMaxSeqLength, int *kvMaxSeqLength, int *maxBatchSize, int *maxBeamSize)</code><br>
Retrieves settings from the previously created attention descriptor.
</td>
</tr>
<tr>
<td>2313</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, size_t *weightSizeInBytes, size_t *workSpaceSizeInBytes, size_t *reserveSpaceSizeInBytes)</code><br>
Computes weight, work, and reserve space buffer sizes used by multi-head attention functions.
</td>
</tr>
<tr>
<td>2314</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes, const void *weights, cudnnTensorDescriptor_t wDesc, void **wAddr)</code><br>
Obtains the shape of the weight or bias tensor.
</td>
</tr>
<tr>
<td>2315</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *maxCount)</code><br>
Retrieves the maximum number of algorithms available for RNN backward weights.
</td>
</tr>
<tr>
<td>2316</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode)</code><br>
Retrieves the RNN bias mode configured by cudnnSetRNNBiasMode().
</td>
</tr>
<tr>
<td>2317</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t RNNDataDesc, cudnnDataType_t *dataType, cudnnRNNDataLayout_t *layout, int *maxSeqLength, int *batchSize, int *vectorSize, int arrayLengthRequested, int seqLengthArray[], void *paddingFill)</code><br>
Retrieves a previously created RNN data descriptor object.
</td>
</tr>
<tr>
<td>2318</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNDescriptor_v6(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int *hiddenSize, int *numLayers, cudnnDropoutDescriptor_t *dropoutDesc, cudnnRNNInputMode_t *inputMode, cudnnDirectionMode_t *direction, cudnnRNNMode_t *cellMode, cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec)</code><br>
Retrieves RNN network parameters configured by cudnnSetRNNDescriptor_v6().
</td>
</tr>
<tr>
<td>2319</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNDescriptor_v8(cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t *algo, cudnnRNNMode_t *cellMode, cudnnRNNBiasMode_t *biasMode, cudnnDirectionMode_t *dirMode, cudnnRNNInputMode_t *inputMode, cudnnDataType_t *dataType, cudnnDataType_t *mathPrec, cudnnMathType_t *mathType, int32_t *inputSize, int32_t *hiddenSize, int32_t *projSize, int32_t *numLayers, cudnnDropoutDescriptor_t *dropoutDesc, uint32_t *auxFlags)</code><br>
Retrieves RNN network parameters configured by cudnnSetRNNDescriptor_v8().
</td>
</tr>
<tr>
<td>2320</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *maxCount)</code><br>
Retrieves the maximum number of algorithms available for RNN forward inference.
</td>
</tr>
<tr>
<td>2321</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int pseudoLayer, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc, void **linLayerBias)</code><br>
Obtains a pointer and descriptor of every RNN bias column vector in each pseudo-layer.
</td>
</tr>
<tr>
<td>2322</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int pseudoLayer, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc, void **linLayerMat)</code><br>
Obtains a pointer and descriptor of every RNN weight matrix in each pseudo-layer.
</td>
</tr>
<tr>
<td>2323</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType)</code><br>
Sets the preferred option to use NVIDIA Tensor Cores accelerators.
</td>
</tr>
<tr>
<td>2324</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t *paddingMode)</code><br>
Retrieves the RNN padding mode from the RNN descriptor.
</td>
</tr>
<tr>
<td>2325</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes, cudnnDataType_t dataType)</code><br>
Queries the amount of parameter space required to execute the RNN.
</td>
</tr>
<tr>
<td>2326</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNProjectionLayers(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int *recProjSize, int *outProjSize)</code><br>
Retrieves the current RNN projection parameters.
</td>
</tr>
<tr>
<td>2327</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNTempSpaceSizes(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnForwardMode_t fMode, cudnnRNNDataDescriptor_t xDesc, size_t *workSpaceSize, size_t *reserveSpaceSize)</code><br>
Computes the work and reserve space buffer sizes based on the RNN network geometry.
</td>
</tr>
<tr>
<td>2328</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, size_t *sizeInBytes)</code><br>
Queries the amount of reserved space required for training the RNN.
</td>
</tr>
<tr>
<td>2329</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNWeightParams(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int32_t pseudoLayer, size_t weightSpaceSize, const void *weightSpace, int32_t linLayerID, cudnnTensorDescriptor_t mDesc, void **mAddr, cudnnTensorDescriptor_t bDesc, void **bAddr)</code><br>
Obtains the start address and shape of every RNN weight matrix and bias vector.
</td>
</tr>
<tr>
<td>2330</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNWeightSpaceSize(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, size_t *weightSpaceSize)</code><br>
Reports the required size of the weight space buffer in bytes.
</td>
</tr>
<tr>
<td>2331</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, size_t *sizeInBytes)</code><br>
Queries the amount of work space required to execute the RNN.
</td>
</tr>
<tr>
<td>2332</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t *dataType, int *nbDims, int nbDimsRequested, int dimA[], cudnnSeqDataAxis_t axes[], size_t *seqLengthArraySize, size_t seqLengthSizeRequested, int seqLengthArray[], void *paddingFill)</code><br>
Retrieves settings from a previously created sequence data descriptor.
</td>
</tr>
<tr>
<td>2333</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnMultiHeadAttnForward(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, int currIdx, const int loWinIdx[], const int hiWinIdx[], const int devSeqLengthsQO[], const int devSeqLengthsKV[], const cudnnSeqDataDescriptor_t qDesc, const void *queries, const void *residuals, const cudnnSeqDataDescriptor_t kDesc, const void *keys, const cudnnSeqDataDescriptor_t vDesc, const void *values, const cudnnSeqDataDescriptor_t oDesc, void *out, size_t weightSizeInBytes, const void *weights, size_t workSpaceSizeInBytes, void *workSpace, size_t reserveSpaceSizeInBytes, void *reserveSpace)</code><br>
Computes the forward responses of the multi-head attention layer.
</td>
</tr>
<tr>
<td>2334</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNForward(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnForwardMode_t fwdMode, const int32_t devSeqLengths[], cudnnRNNDataDescriptor_t xDesc, const void *x, cudnnRNNDataDescriptor_t yDesc, void *y, cudnnTensorDescriptor_t hDesc, const void *hx, void *hy, cudnnTensorDescriptor_t cDesc, const void *cx, void *cy, size_t weightSpaceSize, const void *weightSpace, size_t workSpaceSize, void *workSpace, size_t reserveSpaceSize, void *reserveSpace)</code><br>
Computes the forward response of the recurrent neural network.
</td>
</tr>
<tr>
<td>2335</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes)</code><br>
Executes the recurrent neural network with inputs x, hx, and cx, weights w and outputs y, hy, and cy.
</td>
</tr>
<tr>
<td>2336</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNForwardInferenceEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnRNNDataDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, const cudnnRNNDataDescriptor_t kDesc, const void *keys, const cudnnRNNDataDescriptor_t cDesc, void *cAttn, const cudnnRNNDataDescriptor_t iDesc, void *iAttn, const cudnnRNNDataDescriptor_t qDesc, void *queries, void *workSpace, size_t workSpaceSizeInBytes)</code><br>
Extended version of the RNN forward inference function.
</td>
</tr>
<tr>
<td>2337</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNGetClip(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t *clipMode, cudnnNanPropagation_t *clipNanOpt, double *lclip, double *rclip)</code><br>
Retrieves the current LSTM cell clipping parameters.
</td>
</tr>
<tr>
<td>2338</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNGetClip_v8(cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t *clipMode, cudnnNanPropagation_t *clipNanOpt, double *lclip, double *rclip)</code><br>
Retrieves the current LSTM cell clipping parameters.
</td>
</tr>
<tr>
<td>2339</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNSetClip(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t clipMode, cudnnNanPropagation_t clipNanOpt, double lclip, double rclip)</code><br>
Sets the LSTM cell clipping mode.
</td>
</tr>
<tr>
<td>2340</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNSetClip_v8(cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t clipMode, cudnnNanPropagation_t clipNanOpt, double lclip, double rclip)</code><br>
Sets the LSTM cell clipping mode.
</td>
</tr>
<tr>
<td>2341</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetAttnDescriptor(cudnnAttnDescriptor_t attnDesc, unsigned attnMode, int nHeads, double smScaler, cudnnDataType_t dataType, cudnnDataType_t computePrec, cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc, cudnnDropoutDescriptor_t postDropoutDesc, int qSize, int kSize, int vSize, int qProjSize, int kProjSize, int vProjSize, int oProjSize, int qoMaxSeqLength, int kvMaxSeqLength, int maxBatchSize, int maxBeamSize)</code><br>
Configures a multi-head attention descriptor.
</td>
</tr>
<tr>
<td>2342</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan)</code><br>
Sets the persistent RNN plan to be executed.
</td>
</tr>
<tr>
<td>2343</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetRNNAlgorithmDescriptor(cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t algo)</code><br>
Sets the RNN algorithm descriptor.
</td>
</tr>
<tr>
<td>2344</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode)</code><br>
Sets the number of bias vectors for a previously created RNN descriptor.
</td>
</tr>
<tr>
<td>2345</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t RNNDataDesc, cudnnDataType_t dataType, cudnnRNNDataLayout_t layout, int maxSeqLength, int batchSize, int vectorSize, const int seqLengthArray[], void *paddingFill)</code><br>
Initializes a previously created RNN data descriptor object.
</td>
</tr>
<tr>
<td>2346</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetRNNDescriptor_v6(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize, const int numLayers, cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec)</code><br>
Initializes a previously created RNN descriptor object.
</td>
</tr>
<tr>
<td>2347</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetRNNDescriptor_v8(cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t algo, cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode, cudnnDirectionMode_t dirMode, cudnnRNNInputMode_t inputMode, cudnnDataType_t dataType, cudnnDataType_t mathPrec, cudnnMathType_t mathType, int32_t inputSize, int32_t hiddenSize, int32_t projSize, int32_t numLayers, cudnnDropoutDescriptor_t dropoutDesc, uint32_t auxFlags)</code><br>
Initializes a previously created RNN descriptor object.
</td>
</tr>
<tr>
<td>2348</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType)</code><br>
Sets the preferred option to use NVIDIA Tensor Cores accelerators.
</td>
</tr>
<tr>
<td>2349</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t paddingMode)</code><br>
Enables or disables the padded RNN input/output for a previously created RNN descriptor.
</td>
</tr>
<tr>
<td>2350</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetRNNProjectionLayers(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int recProjSize, int outProjSize)</code><br>
Enables the recurrent and/or output projection in a recursive neural network.
</td>
</tr>
<tr>
<td>2351</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType, int nbDims, const int dimA[], const cudnnSeqDataAxis_t axes[], size_t seqLengthArraySize, const int seqLengthArray[], void *paddingFill)</code><br>
Initializes a previously created sequence data descriptor object.
</td>
</tr>
<tr>
<td>2352</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
