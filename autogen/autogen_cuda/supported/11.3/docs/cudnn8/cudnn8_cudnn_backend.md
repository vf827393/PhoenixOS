<h2>PhOS Support: cuDNN 8.0 - Backend API (0/7)</h2>

<p>
Documentation: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-897/api/index.html#cudnn-backend-so-api

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor)</code><br>
Allocates memory in the descriptor for a given descriptor type.
</td>
</tr>
<tr>
<td>2500</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t descriptor)</code><br>
Destroys instances of cudnnBackendDescriptor_t that were previously created using cudnnBackendCreateDescriptor().
</td>
</tr>
<tr>
<td>2501</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBackendExecute(cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan, cudnnBackendDescriptor_t variantPack)</code><br>
Executes the given Engine Configuration Plan on the VariantPack and the finalized ExecutionPlan on the data.
</td>
</tr>
<tr>
<td>2502</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor)</code><br>
Finalizes the memory pointed to by the descriptor.
</td>
</tr>
<tr>
<td>2503</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBackendGetAttribute(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName, cudnnBackendAttributeType_t attributeType, int64_t requestedElementCount, int64_t *elementCount, void *arrayOfElements)</code><br>
Retrieves the value(s) of an attribute of a descriptor.
</td>
</tr>
<tr>
<td>2504</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBackendInitialize(cudnnBackendDescriptor_t descriptor, cudnnBackendDescriptorType_t descriptorType, size_t sizeInBytes)</code><br>
Repurposes a pre-allocated memory pointed to by a descriptor to a backend descriptor of type descriptorType.
</td>
</tr>
<tr>
<td>2505</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>cudnnStatus_t cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName, cudnnBackendAttributeType_t attributeType, int64_t elementCount, void *arrayOfElements)</code><br>
Sets an attribute of a descriptor to value(s) provided as a pointer.
</td>
</tr>
<tr>
<td>2506</td>
<td>✗</td>
<td>✗</td>
</tr>

</table>
