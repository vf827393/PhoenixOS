<h2>PhOS Support: cuDNN 8.0 - CNN Training (0/20)</h2>

<p>
Documentation: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/api/index.html#cudnn-ops-train-so-api

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCnnTrainVersionCheck ( cudnnHandle_t handle )</code><br>
Checks the version of cuDNN CNN training.
</td>
</tr>
<tr>
<td>2200</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnConvolutionBackwardBias ( cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta, const cudnnTensorDescriptor_t dbDesc, void *db )</code><br>
Computes the gradient of the bias for a convolution.
</td>
</tr>
<tr>
<td>2201</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnConvolutionBackwardFilter ( cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnFilterDescriptor_t dwDesc, void *dw )</code><br>
Computes the gradient of the filter for a convolution.
</td>
</tr>
<tr>
<td>2202</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateFusedOpsConstParamPack ( cudnnFusedOpsConstParamPack_t *constPack )</code><br>
Creates a constant parameter pack for fused operations.
</td>
</tr>
<tr>
<td>2203</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateFusedOpsPlan ( cudnnFusedOpsPlan_t *plan )</code><br>
Creates a plan for fused operations.
</td>
</tr>
<tr>
<td>2204</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateFusedOpsVariantParamPack ( cudnnFusedOpsVariantParamPack_t *varPack )</code><br>
Creates a variant parameter pack for fused operations.
</td>
</tr>
<tr>
<td>2205</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyFusedOpsConstParamPack ( cudnnFusedOpsConstParamPack_t constPack )</code><br>
Destroys a constant parameter pack for fused operations.
</td>
</tr>
<tr>
<td>2206</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyFusedOpsPlan ( cudnnFusedOpsPlan_t plan )</code><br>
Destroys a plan for fused operations.
</td>
</tr>
<tr>
<td>2207</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack ( cudnnFusedOpsVariantParamPack_t varPack )</code><br>
Destroys a variant parameter pack for fused operations.
</td>
</tr>
<tr>
<td>2208</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm ( cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults )</code><br>
Finds the best algorithm for convolution backward filter.
</td>
</tr>
<tr>
<td>2209</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx ( cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, void *dw, int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace, size_t workSpaceSizeInBytes )</code><br>
Finds the best algorithm for convolution backward filter with additional workspace.
</td>
</tr>
<tr>
<td>2210</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnFusedOpsExecute ( cudnnHandle_t handle, cudnnFusedOpsPlan_t plan, cudnnFusedOpsVariantParamPack_t varPack )</code><br>
Executes a fused operation plan.
</td>
</tr>
<tr>
<td>2211</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount ( cudnnHandle_t handle, int *count )</code><br>
Gets the maximum number of algorithms for convolution backward filter.
</td>
</tr>
<tr>
<td>2212</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7 ( cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults )</code><br>
Gets the best algorithm for convolution backward filter using version 7 API.
</td>
</tr>
<tr>
<td>2213</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize ( cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes )</code><br>
Gets the workspace size for convolution backward filter.
</td>
</tr>
<tr>
<td>2214</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute ( cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel, cudnnFusedOpsPointerPlaceHolder_t placeHolder, void *param, size_t *sizeInBytes )</code><br>
Gets an attribute from a constant parameter pack for fused operations.
</td>
</tr>
<tr>
<td>2215</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute ( cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel, cudnnFusedOpsPointerPlaceHolder_t placeHolder, void *param, size_t *sizeInBytes )</code><br>
Gets an attribute from a variant parameter pack for fused operations.
</td>
</tr>
<tr>
<td>2216</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnMakeFusedOpsPlan ( cudnnHandle_t handle, cudnnFusedOpsPlan_t plan, cudnnFusedOpsConstParamPack_t constPack )</code><br>
Creates a plan for fused operations using a constant parameter pack.
</td>
</tr>
<tr>
<td>2217</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute ( cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel, cudnnFusedOpsPointerPlaceHolder_t placeHolder, const void *param, size_t sizeInBytes )</code><br>
Sets an attribute in a constant parameter pack for fused operations.
</td>
</tr>
<tr>
<td>2218</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute ( cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel, cudnnFusedOpsPointerPlaceHolder_t placeHolder, const void *param, size_t sizeInBytes )</code><br>
Sets an attribute in a variant parameter pack for fused operations.
</td>
</tr>
<tr>
<td>2219</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
