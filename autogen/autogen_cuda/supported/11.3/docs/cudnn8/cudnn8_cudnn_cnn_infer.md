<h2>PhOS Support: cuDNN 8.0 - CNN Inference (0/31)</h2>

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
<code>cudnnStatus_t cudnnCnnInferVersionCheck ( cudnnHandle_t handle )</code><br>
Checks the version of cuDNN CNN inference.
</td>
</tr>
<tr>
<td>2100</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnConvolutionBackwardData ( cudnnHandle_t handle, const void *alpha, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx )</code><br>
Computes the gradient of the input data for a convolution.
</td>
</tr>
<tr>
<td>2101</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnConvolutionBiasActivationForward ( cudnnHandle_t handle, const void *alpha1, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *alpha2, const cudnnTensorDescriptor_t zDesc, const void *z, const cudnnTensorDescriptor_t biasDesc, const void *bias, const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void *y )</code><br>
Performs convolution, bias addition, and activation in a single operation.
</td>
</tr>
<tr>
<td>2102</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnConvolutionForward ( cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y )</code><br>
Performs forward convolution.
</td>
</tr>
<tr>
<td>2103</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnCreateConvolutionDescriptor ( cudnnConvolutionDescriptor_t *convDesc )</code><br>
Creates a convolution descriptor.
</td>
</tr>
<tr>
<td>2104</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnDestroyConvolutionDescriptor ( cudnnConvolutionDescriptor_t convDesc )</code><br>
Destroys a convolution descriptor.
</td>
</tr>
<tr>
<td>2105</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm ( cudnnHandle_t handle, const cudnnTensorDescriptor_t dyDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults )</code><br>
Finds the best algorithm for convolution backward data.
</td>
</tr>
<tr>
<td>2106</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx ( cudnnHandle_t handle, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, void *dx, int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace, size_t workSpaceSizeInBytes )</code><br>
Finds the best algorithm for convolution backward data with extended options.
</td>
</tr>
<tr>
<td>2107</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnFindConvolutionForwardAlgorithm ( cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults )</code><br>
Finds the best algorithm for convolution forward.
</td>
</tr>
<tr>
<td>2108</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx ( cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, void *y, int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace, size_t workSpaceSizeInBytes )</code><br>
Finds the best algorithm for convolution forward with extended options.
</td>
</tr>
<tr>
<td>2109</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolution2dDescriptor ( const cudnnConvolutionDescriptor_t convDesc, int *pad_h, int *pad_w, int *u, int *v, int *dilation_h, int *dilation_w, cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType )</code><br>
Gets the 2D convolution descriptor.
</td>
</tr>
<tr>
<td>2110</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolution2dForwardOutputDim ( const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc, const cudnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w )</code><br>
Gets the output dimensions for a 2D convolution.
</td>
</tr>
<tr>
<td>2111</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount ( cudnnHandle_t handle, int *count )</code><br>
Gets the maximum number of algorithms for convolution backward data.
</td>
</tr>
<tr>
<td>2112</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7 ( cudnnHandle_t handle, const cudnnTensorDescriptor_t dyDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults )</code><br>
Gets the best algorithm for convolution backward data using version 7 API.
</td>
</tr>
<tr>
<td>2113</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize ( cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo, size_t *sizeInBytes )</code><br>
Gets the workspace size for convolution backward data.
</td>
</tr>
<tr>
<td>2114</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount ( cudnnHandle_t handle, int *count )</code><br>
Gets the maximum number of algorithms for convolution forward.
</td>
</tr>
<tr>
<td>2115</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7 ( cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults )</code><br>
Gets the best algorithm for convolution forward using version 7 API.
</td>
</tr>
<tr>
<td>2116</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize ( cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo, size_t *sizeInBytes )</code><br>
Gets the workspace size for convolution forward.
</td>
</tr>
<tr>
<td>2117</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionGroupCount ( const cudnnConvolutionDescriptor_t convDesc, int *groupCount )</code><br>
Gets the group count for a convolution descriptor.
</td>
</tr>
<tr>
<td>2118</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionMathType ( const cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType )</code><br>
Gets the math type for a convolution descriptor.
</td>
</tr>
<tr>
<td>2119</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionNdDescriptor ( const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested, int *arrayLength, int padA[], int strideA[], int dilationA[], cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType )</code><br>
Gets the N-dimensional convolution descriptor.
</td>
</tr>
<tr>
<td>2120</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim ( const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc, const cudnnFilterDescriptor_t filterDesc, int nbDims, int tensorOuputDimA[] )</code><br>
Gets the output dimensions for an N-dimensional convolution.
</td>
</tr>
<tr>
<td>2121</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetConvolutionReorderType ( const cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType )</code><br>
Gets the reorder type for a convolution descriptor.
</td>
</tr>
<tr>
<td>2122</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors ( cudnnHandle_t handle, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dxDesc, cudnnTensorDescriptor_t foldedDyDesc, cudnnFilterDescriptor_t foldedWDesc, cudnnTensorDescriptor_t foldedDxDesc )</code><br>
Gets the folded descriptors for convolution backward data.
</td>
</tr>
<tr>
<td>2123</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnIm2Col ( cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, void *colBuffer )</code><br>
Performs the im2col operation.
</td>
</tr>
<tr>
<td>2124</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnReorderFilterAndBias ( cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, cudnnReorderType_t reorderType, const void *w, void *wReordered, const cudnnTensorDescriptor_t biasDesc, const void *bias, void *biasReordered )</code><br>
Reorders filter and bias.
</td>
</tr>
<tr>
<td>2125</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetConvolution2dDescriptor ( cudnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u, int v, int dilation_h, int dilation_w, cudnnConvolutionMode_t mode, cudnnDataType_t computeType )</code><br>
Sets the 2D convolution descriptor.
</td>
</tr>
<tr>
<td>2126</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetConvolutionGroupCount ( cudnnConvolutionDescriptor_t convDesc, int groupCount )</code><br>
Sets the group count for a convolution descriptor.
</td>
</tr>
<tr>
<td>2127</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetConvolutionMathType ( cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType )</code><br>
Sets the math type for a convolution descriptor.
</td>
</tr>
<tr>
<td>2128</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetConvolutionNdDescriptor ( cudnnConvolutionDescriptor_t convDesc, int arrayLength, const int padA[], const int strideA[], const int dilationA[], cudnnConvolutionMode_t mode, cudnnDataType_t computeType )</code><br>
Sets the N-dimensional convolution descriptor.
</td>
</tr>
<tr>
<td>2129</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnSetConvolutionReorderType ( cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType )</code><br>
Sets the reorder type for a convolution descriptor.
</td>
</tr>
<tr>
<td>2130</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
