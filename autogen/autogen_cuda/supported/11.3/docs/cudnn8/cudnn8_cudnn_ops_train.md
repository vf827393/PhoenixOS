<h2>PhOS Support: cuDNN 8.0 - Operations Training (0/21)</h2>

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
<code>cudnnStatus_t cudnnActivationBackward ( cudnnHandle_t handle, const cudnnActivationDescriptor_t activationDesc, const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx )</code><br>
Performs backward activation.
</td>
</tr>
<tr>
<td>2000</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBatchNormalizationBackward ( cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alphaDataDiff, const void *betaDataDiff, const void *alphaParamDiff, const void *betaParamDiff, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t dxDesc, void *dx, const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale, const void *bnBias, void *dBnScaleResult, void *dBnBiasResult, double epsilon, const void *savedMean, const void *savedInvVariance )</code><br>
Performs batch normalization backward.
</td>
</tr>
<tr>
<td>2001</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBatchNormalizationBackwardEx ( cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alphaDataDiff, const void *betaDataDiff, const void *alphaParamDiff, const void *betaParamDiff, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t yDesc, const void *yData, const cudnnTensorDescriptor_t dyDesc, const void *dyData, const cudnnTensorDescriptor_t dzDesc, void *dzData, const cudnnTensorDescriptor_t dxDesc, void *dxData, const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScaleData, const void *bnBiasData, void *dBnScaleData, void *dBnBiasData, double epsilon, const void *savedMean, const void *savedInvVariance, void *activationDesc, void *workSpace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes )</code><br>
Performs batch normalization backward with extended options.
</td>
</tr>
<tr>
<td>2002</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBatchNormalizationForwardTraining ( cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, double exponentialAverageFactor, void *resultRunningMean, void *resultRunningVariance, double epsilon, void *resultSaveMean, void *resultSaveInvVariance )</code><br>
Performs batch normalization forward training.
</td>
</tr>
<tr>
<td>2003</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx ( cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc, const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMean, void *resultRunningVariance, double epsilon, void *resultSaveMean, void *resultSaveInvVariance, void *activationDesc, void *workSpace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes )</code><br>
Performs batch normalization forward training with extended options.
</td>
</tr>
<tr>
<td>2004</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDivisiveNormalizationBackward ( cudnnHandle_t handle, cudnnLRNMode_t mode, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t dxDesc, void *dx, const cudnnTensorDescriptor_t dDbnScaleBiasDesc, const void *dbnScale, const void *dbnBias, void *dDbnScaleResult, void *dDbnBiasResult, double epsilon, const void *savedMean, const void *savedInvVariance )</code><br>
Performs divisive normalization backward.
</td>
</tr>
<tr>
<td>2005</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDropoutBackward ( cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t dxDesc, void *dx, const void *reserveSpace, size_t reserveSpaceSizeInBytes )</code><br>
Performs dropout backward.
</td>
</tr>
<tr>
<td>2006</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize ( cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc, const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dBnScaleBiasDesc, size_t *sizeInBytes )</code><br>
Gets workspace size for batch normalization backward with extended options.
</td>
</tr>
<tr>
<td>2007</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize ( cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc, const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, size_t *sizeInBytes )</code><br>
Gets workspace size for batch normalization forward training with extended options.
</td>
</tr>
<tr>
<td>2008</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize ( cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc, const cudnnTensorDescriptor_t yDesc, size_t *sizeInBytes )</code><br>
Gets reserve space size for batch normalization training with extended options.
</td>
</tr>
<tr>
<td>2009</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetNormalizationBackwardWorkspaceSize ( cudnnHandle_t handle, cudnnNormMode_t mode, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dBnScaleBiasDesc, size_t *sizeInBytes )</code><br>
Gets workspace size for normalization backward.
</td>
</tr>
<tr>
<td>2010</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetNormalizationForwardTrainingWorkspaceSize ( cudnnHandle_t handle, cudnnNormMode_t mode, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, size_t *sizeInBytes )</code><br>
Gets workspace size for normalization forward training.
</td>
</tr>
<tr>
<td>2011</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetNormalizationTrainingReserveSpaceSize ( cudnnHandle_t handle, cudnnNormMode_t mode, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc, size_t *sizeInBytes )</code><br>
Gets reserve space size for normalization training.
</td>
</tr>
<tr>
<td>2012</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnLRNCrossChannelBackward ( cudnnHandle_t handle, cudnnLRNMode_t mode, const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx )</code><br>
Performs LRN cross-channel backward.
</td>
</tr>
<tr>
<td>2013</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnNormalizationBackward ( cudnnHandle_t handle, cudnnNormMode_t mode, const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx, const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale, const void *bnBias, void *dBnScaleResult, void *dBnBiasResult, double epsilon, const void *savedMean, const void *savedInvVariance )</code><br>
Performs normalization backward.
</td>
</tr>
<tr>
<td>2014</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnNormalizationForwardTraining ( cudnnHandle_t handle, cudnnNormMode_t mode, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, double exponentialAverageFactor, void *resultRunningMean, void *resultRunningVariance, double epsilon, void *resultSaveMean, void *resultSaveInvVariance )</code><br>
Performs normalization forward training.
</td>
</tr>
<tr>
<td>2015</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnOpsTrainVersionCheck ( cudnnHandle_t handle )</code><br>
Checks the version of cuDNN operations training.
</td>
</tr>
<tr>
<td>2016</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnPoolingBackward ( cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc, const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx )</code><br>
Performs pooling backward.
</td>
</tr>
<tr>
<td>2017</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSoftmaxBackward ( cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode, const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx )</code><br>
Performs softmax backward.
</td>
</tr>
<tr>
<td>2018</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSpatialTfGridGeneratorBackward ( cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc, const void *dgrid, void *dtheta )</code><br>
Generates a grid for spatial transformer backward.
</td>
</tr>
<tr>
<td>2019</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSpatialTfSamplerBackward ( cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx, const void *grid, const cudnnTensorDescriptor_t dyDesc, const void *dy )</code><br>
Performs spatial transformer sampling backward.
</td>
</tr>
<tr>
<td>2020</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
