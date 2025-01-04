
# [Graph Management (0/72)](https://docs.nvidia.com/cuda/archive/11.3.0/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)

<table>
<tr>
<th>Supported</th>
<th>Test Passed</th>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddChildGraphNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph )</code><br>
Creates a child graph node and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddDependencies ( cudaGraph_t graph, const cudaGraphNode_t* from, const cudaGraphNode_t* to, size_t numDependencies )</code><br>
Adds dependency edges to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddEmptyNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies )</code><br>
Creates an empty node and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddEventRecordNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event )</code><br>
Creates an event record node and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddEventWaitNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event )</code><br>
Creates an event wait node and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddExternalSemaphoresSignalNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams )</code><br>
Creates an external semaphore signal node and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddExternalSemaphoresWaitNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams )</code><br>
Creates an external semaphore wait node and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddHostNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams )</code><br>
Creates a host execution node and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddKernelNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams )</code><br>
Creates a kernel execution node and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddMemcpyNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams )</code><br>
Creates a memcpy node and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddMemcpyNode1D ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind )</code><br>
Creates a 1D memcpy node and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddMemcpyNodeFromSymbol ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind )</code><br>
Creates a memcpy node to copy from a symbol on the device and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddMemcpyNodeToSymbol ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind )</code><br>
Creates a memcpy node to copy to a symbol on the device and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphAddMemsetNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams )</code><br>
Creates a memset node and adds it to a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphChildGraphNodeGetGraph ( cudaGraphNode_t node, cudaGraph_t* pGraph )</code><br>
Gets a handle to the embedded graph of a child graph node.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphClone ( cudaGraph_t* pGraphClone, cudaGraph_t originalGraph )</code><br>
Clones a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphCreate ( cudaGraph_t* pGraph, unsigned int  flags )</code><br>
Creates a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphDebugDotPrint ( cudaGraph_t graph, const char* path, unsigned int  flags )</code><br>
Write a DOT file describing graph structure.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphDestroy ( cudaGraph_t graph )</code><br>
Destroys a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphDestroyNode ( cudaGraphNode_t node )</code><br>
Remove a node from the graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphEventRecordNodeGetEvent ( cudaGraphNode_t node, cudaEvent_t* event_out )</code><br>
Returns the event associated with an event record node.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphEventRecordNodeSetEvent ( cudaGraphNode_t node, cudaEvent_t event )</code><br>
Sets an event record node's event.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphEventWaitNodeGetEvent ( cudaGraphNode_t node, cudaEvent_t* event_out )</code><br>
Returns the event associated with an event wait node.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphEventWaitNodeSetEvent ( cudaGraphNode_t node, cudaEvent_t event )</code><br>
Sets an event wait node's event.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecChildGraphNodeSetParams ( cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph )</code><br>
Updates node parameters in the child graph node in the given graphExec.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecDestroy ( cudaGraphExec_t graphExec )</code><br>
Destroys an executable graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecEventRecordNodeSetEvent ( cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event )</code><br>
Sets the event for an event record node in the given graphExec.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecEventWaitNodeSetEvent ( cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event )</code><br>
Sets the event for an event wait node in the given graphExec.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams ( cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams )</code><br>
Sets the parameters for an external semaphore signal node in the given graphExec.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams ( cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams )</code><br>
Sets the parameters for an external semaphore wait node in the given graphExec.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecHostNodeSetParams ( cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams )</code><br>
Sets the parameters for a host node in the given graphExec.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecKernelNodeSetParams ( cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams )</code><br>
Sets the parameters for a kernel node in the given graphExec.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecMemcpyNodeSetParams ( cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams )</code><br>
Sets the parameters for a memcpy node in the given graphExec.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecMemcpyNodeSetParams1D ( cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind )</code><br>
Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional copy.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol ( cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind )</code><br>
Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol ( cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind )</code><br>
Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecMemsetNodeSetParams ( cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams )</code><br>
Sets the parameters for a memset node in the given graphExec.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExecUpdate ( cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t* hErrorNode_out, cudaGraphExecUpdateResult ** updateResult_out )</code><br>
Check whether an executable graph can be updated with a graph and perform the update if possible.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams ( cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out )</code><br>
Returns an external semaphore signal node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams ( cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams )</code><br>
Sets an external semaphore signal node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams ( cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out )</code><br>
Returns an external semaphore wait node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams ( cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams )</code><br>
Sets an external semaphore wait node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphGetEdges ( cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t* numEdges )</code><br>
Returns a graph's dependency edges.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphGetNodes ( cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes )</code><br>
Returns a graph's nodes.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphGetRootNodes ( cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes )</code><br>
Returns a graph's root nodes.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphHostNodeGetParams ( cudaGraphNode_t node, cudaHostNodeParams* pNodeParams )</code><br>
Returns a host node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphHostNodeSetParams ( cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams )</code><br>
Sets a host node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphInstantiate ( cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize )</code><br>
Creates an executable graph from a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphKernelNodeCopyAttributes ( cudaGraphNode_t hSrc, cudaGraphNode_t hDst )</code><br>
Copies attributes from source node to destination node.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphKernelNodeGetAttribute ( cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out )</code><br>
Queries node attribute.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphKernelNodeGetParams ( cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams )</code><br>
Returns a kernel node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphKernelNodeSetAttribute ( cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value )</code><br>
Sets node attribute.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphKernelNodeSetParams ( cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams )</code><br>
Sets a kernel node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphLaunch ( cudaGraphExec_t graphExec, cudaStream_t stream )</code><br>
Launches an executable graph in a stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphMemcpyNodeGetParams ( cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams )</code><br>
Returns a memcpy node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphMemcpyNodeSetParams ( cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams )</code><br>
Sets a memcpy node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<th>Supported</th>
<th>Test Passed</th>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphMemcpyNodeSetParams1D ( cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind )</code><br>
Sets a memcpy node's parameters to perform a 1-dimensional copy.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol ( cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind )</code><br>
Sets a memcpy node's parameters to copy from a symbol on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol ( cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind )</code><br>
Sets a memcpy node's parameters to copy to a symbol on the device.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphMemsetNodeGetParams ( cudaGraphNode_t node, cudaMemsetParams* pNodeParams )</code><br>
Returns a memset node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphMemsetNodeSetParams ( cudaGraphNode_t node, const cudaMemsetParams* pNodeParams )</code><br>
Sets a memset node's parameters.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphNodeFindInClone ( cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph )</code><br>
Finds a cloned version of a node.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphNodeGetDependencies ( cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies )</code><br>
Returns a node's dependencies.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphNodeGetDependentNodes ( cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes )</code><br>
Returns a node's dependent nodes.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphNodeGetType ( cudaGraphNode_t node, cudaGraphNodeType ** pType )</code><br>
Returns a node's type.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphReleaseUserObject ( cudaGraph_t graph, cudaUserObject_t object, unsigned int count = 1 )</code><br>
Release a user object reference from a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphRemoveDependencies ( cudaGraph_t graph, const cudaGraphNode_t* from, const cudaGraphNode_t* to, size_t numDependencies )</code><br>
Removes dependency edges from a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphRetainUserObject ( cudaGraph_t graph, cudaUserObject_t object, unsigned int count = 1, unsigned int flags = 0 )</code><br>
Retain a reference to a user object from a graph.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaGraphUpload ( cudaGraphExec_t graphExec, cudaStream_t stream )</code><br>
Uploads an executable graph in a stream.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaUserObjectCreate ( cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags )</code><br>
Create a user object.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaUserObjectRelease ( cudaUserObject_t object, unsigned int count = 1 )</code><br>
Release a reference to a user object.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<td colspan=2>
<code>cudaError_t cudaUserObjectRetain ( cudaUserObject_t object, unsigned int count = 1 )</code><br>
Retain a reference to a user object.
</td>
</tr>
<tr>
<td>✗</td>
<td>✗</td>
</tr>
</table>


