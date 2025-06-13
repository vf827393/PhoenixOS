<h2>PhOS Support: CUDA 11.3 - Driver APIs - Graph Management (0/63)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__GRAPH.html

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphAddChildGraphNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph )</code><br>
Creates a child graph node and adds it to a graph.
</td>
</tr>
<tr>
<td>930</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphAddDependencies ( CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies )</code><br>
Adds dependency edges to a graph.
</td>
</tr>
<tr>
<td>931</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphAddEmptyNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies )</code><br>
Creates an empty node and adds it to a graph.
</td>
</tr>
<tr>
<td>932</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphAddEventRecordNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event )</code><br>
Creates an event record node and adds it to a graph.
</td>
</tr>
<tr>
<td>933</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphAddEventWaitNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event )</code><br>
Creates an event wait node and adds it to a graph.
</td>
</tr>
<tr>
<td>934</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphAddExternalSemaphoresSignalNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams )</code><br>
Creates an external semaphore signal node and adds it to a graph.
</td>
</tr>
<tr>
<td>935</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphAddExternalSemaphoresWaitNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams )</code><br>
Creates an external semaphore wait node and adds it to a graph.
</td>
</tr>
<tr>
<td>936</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphAddHostNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams )</code><br>
Creates a host execution node and adds it to a graph.
</td>
</tr>
<tr>
<td>937</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphAddKernelNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams )</code><br>
Creates a kernel execution node and adds it to a graph.
</td>
</tr>
<tr>
<td>938</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphAddMemcpyNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx )</code><br>
Creates a memcpy node and adds it to a graph.
</td>
</tr>
<tr>
<td>939</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphAddMemsetNode ( CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx )</code><br>
Creates a memset node and adds it to a graph.
</td>
</tr>
<tr>
<td>940</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphChildGraphNodeGetGraph ( CUgraphNode hNode, CUgraph* phGraph )</code><br>
Gets a handle to the embedded graph of a child graph node.
</td>
</tr>
<tr>
<td>941</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphClone ( CUgraph* phGraphClone, CUgraph originalGraph )</code><br>
Clones a graph.
</td>
</tr>
<tr>
<td>942</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphCreate ( CUgraph* phGraph, unsigned int  flags )</code><br>
Creates a graph.
</td>
</tr>
<tr>
<td>943</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphDebugDotPrint ( CUgraph hGraph, const char* path, unsigned int  flags )</code><br>
Write a DOT file describing graph structure.
</td>
</tr>
<tr>
<td>944</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphDestroy ( CUgraph hGraph )</code><br>
Destroys a graph.
</td>
</tr>
<tr>
<td>945</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphDestroyNode ( CUgraphNode hNode )</code><br>
Remove a node from the graph.
</td>
</tr>
<tr>
<td>946</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphEventRecordNodeGetEvent ( CUgraphNode hNode, CUevent* event_out )</code><br>
Returns the event associated with an event record node.
</td>
</tr>
<tr>
<td>947</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphEventRecordNodeSetEvent ( CUgraphNode hNode, CUevent event )</code><br>
Sets an event record node's event.
</td>
</tr>
<tr>
<td>948</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphEventWaitNodeGetEvent ( CUgraphNode hNode, CUevent* event_out )</code><br>
Returns the event associated with an event wait node.
</td>
</tr>
<tr>
<td>949</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphEventWaitNodeSetEvent ( CUgraphNode hNode, CUevent event )</code><br>
Sets an event wait node's event.
</td>
</tr>
<tr>
<td>950</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExecChildGraphNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph )</code><br>
Updates node parameters in the child graph node in the given graphExec.
</td>
</tr>
<tr>
<td>951</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExecDestroy ( CUgraphExec hGraphExec )</code><br>
Destroys an executable graph.
</td>
</tr>
<tr>
<td>952</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExecEventRecordNodeSetEvent ( CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event )</code><br>
Sets the event for an event record node in the given graphExec.
</td>
</tr>
<tr>
<td>953</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExecEventWaitNodeSetEvent ( CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event )</code><br>
Sets the event for an event wait node in the given graphExec.
</td>
</tr>
<tr>
<td>954</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams )</code><br>
Sets the parameters for an external semaphore signal node in the given graphExec.
</td>
</tr>
<tr>
<td>955</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams )</code><br>
Sets the parameters for an external semaphore wait node in the given graphExec.
</td>
</tr>
<tr>
<td>956</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExecHostNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams )</code><br>
Sets the parameters for a host node in the given graphExec.
</td>
</tr>
<tr>
<td>957</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExecKernelNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams )</code><br>
Sets the parameters for a kernel node in the given graphExec.
</td>
</tr>
<tr>
<td>958</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExecMemcpyNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx )</code><br>
Sets the parameters for a memcpy node in the given graphExec.
</td>
</tr>
<tr>
<td>959</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExecMemsetNodeSetParams ( CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx )</code><br>
Sets the parameters for a memset node in the given graphExec.
</td>
</tr>
<tr>
<td>960</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExecUpdate ( CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode* hErrorNode_out, CUgraphExecUpdateResult* updateResult_out )</code><br>
Check whether an executable graph can be updated with a graph and perform the update if possible.
</td>
</tr>
<tr>
<td>961</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExternalSemaphoresSignalNodeGetParams ( CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out )</code><br>
Returns an external semaphore signal node's parameters.
</td>
</tr>
<tr>
<td>962</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExternalSemaphoresSignalNodeSetParams ( CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams )</code><br>
Sets an external semaphore signal node's parameters.
</td>
</tr>
<tr>
<td>963</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExternalSemaphoresWaitNodeGetParams ( CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out )</code><br>
Returns an external semaphore wait node's parameters.
</td>
</tr>
<tr>
<td>964</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphExternalSemaphoresWaitNodeSetParams ( CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams )</code><br>
Sets an external semaphore wait node's parameters.
</td>
</tr>
<tr>
<td>965</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphGetEdges ( CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges )</code><br>
Returns a graph's dependency edges.
</td>
</tr>
<tr>
<td>966</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphGetNodes ( CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes )</code><br>
Returns a graph's nodes.
</td>
</tr>
<tr>
<td>967</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphGetRootNodes ( CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes )</code><br>
Returns a graph's root nodes.
</td>
</tr>
<tr>
<td>968</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphHostNodeGetParams ( CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams )</code><br>
Returns a host node's parameters.
</td>
</tr>
<tr>
<td>969</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphHostNodeSetParams ( CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams )</code><br>
Sets a host node's parameters.
</td>
</tr>
<tr>
<td>970</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphInstantiate ( CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode, char* logBuffer, size_t bufferSize )</code><br>
Creates an executable graph from a graph.
</td>
</tr>
<tr>
<td>971</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphKernelNodeCopyAttributes ( CUgraphNode dst, CUgraphNode src )</code><br>
Copies attributes from source node to destination node.
</td>
</tr>
<tr>
<td>972</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphKernelNodeGetAttribute ( CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out )</code><br>
Queries node attribute.
</td>
</tr>
<tr>
<td>973</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphKernelNodeGetParams ( CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams )</code><br>
Returns a kernel node's parameters.
</td>
</tr>
<tr>
<td>974</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphKernelNodeSetAttribute ( CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value )</code><br>
Sets node attribute.
</td>
</tr>
<tr>
<td>975</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphKernelNodeSetParams ( CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams )</code><br>
Sets a kernel node's parameters.
</td>
</tr>
<tr>
<td>976</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphLaunch ( CUgraphExec hGraphExec, CUstream hStream )</code><br>
Launches an executable graph in a stream.
</td>
</tr>
<tr>
<td>977</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphMemcpyNodeGetParams ( CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams )</code><br>
Returns a memcpy node's parameters.
</td>
</tr>
<tr>
<td>978</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphMemcpyNodeSetParams ( CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams )</code><br>
Sets a memcpy node's parameters.
</td>
</tr>
<tr>
<td>979</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphMemsetNodeGetParams ( CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams )</code><br>
Returns a memset node's parameters.
</td>
</tr>
<tr>
<td>980</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphMemsetNodeSetParams ( CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams )</code><br>
Sets a memset node's parameters.
</td>
</tr>
<tr>
<td>981</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphNodeFindInClone ( CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph )</code><br>
Finds a cloned version of a node.
</td>
</tr>
<tr>
<td>982</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphNodeGetDependencies ( CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies )</code><br>
Returns a node's dependencies.
</td>
</tr>
<tr>
<td>983</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphNodeGetDependentNodes ( CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes )</code><br>
Returns a node's dependent nodes.
</td>
</tr>
<tr>
<td>984</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphNodeGetType ( CUgraphNode hNode, CUgraphNodeType* type )</code><br>
Returns a node's type.
</td>
</tr>
<tr>
<td>985</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphReleaseUserObject ( CUgraph graph, CUuserObject object, unsigned int  count )</code><br>
Release a user object reference from a graph.
</td>
</tr>
<tr>
<td>986</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphRemoveDependencies ( CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies )</code><br>
Removes dependency edges from a graph.
</td>
</tr>
<tr>
<td>987</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphRetainUserObject ( CUgraph graph, CUuserObject object, unsigned int  count, unsigned int  flags )</code><br>
Retain a reference to a user object from a graph.
</td>
</tr>
<tr>
<td>988</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuGraphUpload ( CUgraphExec hGraphExec, CUstream hStream )</code><br>
Uploads an executable graph in a stream.
</td>
</tr>
<tr>
<td>989</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuUserObjectCreate ( CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int  initialRefcount, unsigned int  flags )</code><br>
Create a user object.
</td>
</tr>
<tr>
<td>990</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuUserObjectRelease ( CUuserObject object, unsigned int  count )</code><br>
Release a reference to a user object.
</td>
</tr>
<tr>
<td>991</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuUserObjectRetain ( CUuserObject object, unsigned int  count )</code><br>
Retain a reference to a user object.
</td>
</tr>
<tr>
<td>992</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
