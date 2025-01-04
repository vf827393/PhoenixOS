<h2>PhOS Support: cuDNN 8.0 - Advanced Training (0/28)</h2>

<p>
Documentation: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/api/index.html#cudnn-adv-train-so-api

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnAdvTrainVersionCheck(void)</code><br>
Checks whether the version of the AdvTrain subset of the library is consistent with the other sub-libraries.
</td>
</tr>
<tr>
<td>2400</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t* ctcLossDesc)</code><br>
Creates a CTC loss function descriptor.
</td>
</tr>
<tr>
<td>2401</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCTCLoss(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc, const void *probs, const int hostLabels[], const int hostLabelLengths[], const int hostInputLengths[], void *costs, const cudnnTensorDescriptor_t gradientsDesc, const void *gradients, cudnnCTCLossAlgo_t algo, const cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace, size_t *workSpaceSizeInBytes)</code><br>
Returns the CTC costs and gradients, given the probabilities and labels.
</td>
</tr>
<tr>
<td>2402</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCTCLoss_v8(cudnnHandle_t handle, cudnnCTCLossAlgo_t algo, const cudnnCTCLossDescriptor_t ctcLossDesc, const cudnnTensorDescriptor_t probsDesc, const void *probs, const int labels[], const int labelLengths[], const int inputLengths[], void *costs, const cudnnTensorDescriptor_t gradientsDesc, const void *gradients, size_t *workSpaceSizeInBytes, void *workspace)</code><br>
Returns the CTC costs and gradients, given the probabilities and labels, with support for CUDA graphs.
</td>
</tr>
<tr>
<td>2403</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc)</code><br>
Destroys a CTC loss function descriptor object.
</td>
</tr>
<tr>
<td>2404</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *yDesc, const void *y, const cudnnTensorDescriptor_t *dyDesc, const void *dy, const cudnnTensorDescriptor_t dhyDesc, const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnTensorDescriptor_t *dxDesc, void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx, const cudnnTensorDescriptor_t dcxDesc, void *dcx, const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes, const void *reserveSpace, size_t reserveSpaceSizeInBytes)</code><br>
Attempts all available cuDNN algorithms for cudnnRNNBackwardData(), using user-allocated GPU memory.
</td>
</tr>
<tr>
<td>2405</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t *yDesc, const void *y, const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults, const void *workspace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw, const void *reserveSpace, size_t reserveSpaceSizeInBytes)</code><br>
Attempts all available cuDNN algorithms for cudnnRNNBackwardWeights(), using user-allocated GPU memory.
</td>
</tr>
<tr>
<td>2406</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNBackwardData(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *yDesc, const void *y, const cudnnTensorDescriptor_t *dyDesc, const void *dy, const cudnnTensorDescriptor_t dhyDesc, const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnTensorDescriptor_t *dxDesc, void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx, const cudnnTensorDescriptor_t dcxDesc, void *dcx, void *workspace, size_t workSpaceSizeInBytes, const void *reserveSpace, size_t reserveSpaceSizeInBytes)</code><br>
Executes the recurrent neural network described by rnnDesc with output gradients.
</td>
</tr>
<tr>
<td>2407</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNBackwardData_v8(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int32_t devSeqLengths[], cudnnRNNDataDescriptor_t yDesc, const void *y, const void *dy, cudnnRNNDataDescriptor_t xDesc, void *dx, cudnnTensorDescriptor_t hDesc, const void *hx, const void *dhy, void *dhx, cudnnTensorDescriptor_t cDesc, const void *cx, const void *dcy, void *dcx, size_t weightSpaceSize, const void *weightSpace, size_t workSpaceSize, void *workSpace, size_t reserveSpaceSize, void *reserveSpace)</code><br>
Executes the recurrent neural network described by rnnDesc with output gradients, supporting variable sequence lengths.
</td>
</tr>
<tr>
<td>2408</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNBackwardWeights(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t *yDesc, const void *y, void *workSpace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw, const void *reserveSpace, size_t reserveSpaceSizeInBytes)</code><br>
Accumulates weight gradients from the recurrent neural network described by rnnDesc.
</td>
</tr>
<tr>
<td>2409</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNBackwardWeights_v8(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnWgradMode_t addGrad, const int32_t devSeqLengths[], cudnnRNNDataDescriptor_t xDesc, const void *x, cudnnTensorDescriptor_t hDesc, const void *hx, cudnnRNNDataDescriptor_t yDesc, const void *y, size_t weightSpaceSize, void *dweightSpace, size_t workSpaceSize, void *workSpace, size_t reserveSpaceSize, void *reserveSpace)</code><br>
Computes exact, first-order derivatives of the RNN model with respect to all trainable parameters.
</td>
</tr>
<tr>
<td>2410</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNForwardTraining(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes)</code><br>
Executes the recurrent neural network described by rnnDesc with inputs and outputs.
</td>
</tr>
<tr>
<td>2411</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNForwardTrainingEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnRNNDataDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, const cudnnRNNDataDescriptor_t kDesc, const void *keys, const cudnnRNNDataDescriptor_t cDesc, void *cAttn, const cudnnRNNDataDescriptor_t iDesc, void *iAttn, const cudnnRNNDataDescriptor_t qDesc, void *queries, void *workSpace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes)</code><br>
Executes the recurrent neural network described by rnnDesc with inputs and outputs, supporting unpacked layouts.
</td>
</tr>
<tr>
<td>2412</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType)</code><br>
Sets a CTC loss function descriptor.
</td>
</tr>
<tr>
<td>2413</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType, cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode)</code><br>
Sets a CTC loss function descriptor with additional normalization and gradient propagation options.
</td>
</tr>
<tr>
<td>2414</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType, cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode, int maxLabelLength)</code><br>
Sets a CTC loss function descriptor with support for CUDA graphs.
</td>
</tr>
<tr>
<td>2415</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t* compType)</code><br>
Returns the configuration of the passed CTC loss function descriptor.
</td>
</tr>
<tr>
<td>2416</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t* compType, cudnnLossNormalizationMode_t* normMode, cudnnNanPropagation_t* gradMode)</code><br>
Returns the configuration of the passed CTC loss function descriptor with additional options.
</td>
</tr>
<tr>
<td>2417</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetCTCLossDescriptor_v8(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t* compType, cudnnLossNormalizationMode_t* normMode, cudnnNanPropagation_t* gradMode, int* maxLabelLength)</code><br>
Returns the configuration of the passed CTC loss function descriptor with support for CUDA graphs.
</td>
</tr>
<tr>
<td>2418</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetCTCLossWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc, const cudnnTensorDescriptor_t gradientsDesc, const int* labels, const int* labelLengths, const int* inputLengths, cudnnCTCLossAlgo_t algo, const cudnnCTCLossDescriptor_t ctcLossDesc, size_t* sizeInBytes)</code><br>
Returns the amount of GPU memory workspace needed to call cudnnCTCLoss() with the specified algorithm.
</td>
</tr>
<tr>
<td>2419</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetCTCLossWorkspaceSize_v8(cudnnHandle_t handle, cudnnCTCLossAlgo_t algo, const cudnnCTCLossDescriptor_t ctcLossDesc, const cudnnTensorDescriptor_t probsDesc, const cudnnTensorDescriptor_t gradientsDesc, size_t* sizeInBytes)</code><br>
Returns the amount of GPU memory workspace needed to call cudnnCTCLoss_v8() with the specified algorithm.
</td>
</tr>
<tr>
<td>2420</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount()</code><br>
This function has been deprecated in cuDNN 8.0.
</td>
</tr>
<tr>
<td>2421</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount()</code><br>
This function has been deprecated in cuDNN 8.0.
</td>
</tr>
<tr>
<td>2422</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnMultiHeadAttnBackwardData(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, const int loWinIdx[], const int hiWinIdx[], const int devSeqLengthsDQDO[], const int devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t doDesc, const void *dout, const cudnnSeqDataDescriptor_t dqDesc, void *dqueries, const void *queries, const cudnnSeqDataDescriptor_t dkDesc, void *dkeys, const void *keys, const cudnnSeqDataDescriptor_t dvDesc, void *dvalues, const void *values, size_t weightSizeInBytes, const void *weights, size_t workSpaceSizeInBytes, void *workSpace, size_t reserveSpaceSizeInBytes, void *reserveSpace)</code><br>
Computes exact, first-order derivatives of the multi-head attention block with respect to its inputs: Q, K, V.
</td>
</tr>
<tr>
<td>2423</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, cudnnWgradMode_t addGrad, const cudnnSeqDataDescriptor_t qDesc, const void *queries, const cudnnSeqDataDescriptor_t kDesc, const void *keys, const cudnnSeqDataDescriptor_t vDesc, const void *values, const cudnnSeqDataDescriptor_t doDesc, const void *dout, size_t weightSizeInBytes, const void *weights, void *dweights, size_t workSpaceSizeInBytes, void *workSpace, size_t reserveSpaceSizeInBytes, void *reserveSpace)</code><br>
Computes exact, first-order derivatives of the multi-head attention block with respect to its trainable parameters: projection weights and biases.
</td>
</tr>
<tr>
<td>2424</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNBackwardData(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *yDesc, const void *y, const cudnnTensorDescriptor_t *dyDesc, const void *dy, const cudnnTensorDescriptor_t dhyDesc, const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnTensorDescriptor_t *dxDesc, void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx, const cudnnTensorDescriptor_t dcxDesc, void *dcx, void *workspace, size_t workSpaceSizeInBytes, const void *reserveSpace, size_t reserveSpaceSizeInBytes)</code><br>
This function has been deprecated in cuDNN 8.0. Use cudnnRNNBackwardData_v8() instead.
</td>
</tr>
<tr>
<td>2425</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnRNNBackwardWeights(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t *yDesc, const void *y, void *workSpace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw, void *reserveSpace, size_t reserveSpaceSizeInBytes)</code><br>
This function has been deprecated in cuDNN 8.0. Use cudnnRNNBackwardWeights_v8() instead.
</td>
</tr>
<tr>
<td>2426</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
