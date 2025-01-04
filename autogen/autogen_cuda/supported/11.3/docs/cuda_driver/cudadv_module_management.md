<h2>PhOS Support: CUDA 11.3 - Driver APIs - Module Management (0/14)</h2>

<p>
Documentation: https://docs.nvidia.com/cuda/archive/11.3.0/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE

<table>
<tr>
<th>Index</th>
<th>Supported</th>
<th>Test Passed</th>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuModuleLoad ( CUmodule* module, const char* fname )</code><br>
Loads a compute module.
</td>
</tr>
<tr>
<td>700</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuModuleLoadData ( CUmodule* module, const void* image )</code><br>
Load a module's data.
</td>
</tr>
<tr>
<td>701</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuModuleLoadDataEx ( CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues )</code><br>
Load a module's data with options.
</td>
</tr>
<tr>
<td>702</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuModuleLoadFatBinary ( CUmodule* module, const void* fatCubin )</code><br>
Load a module's data.
</td>
</tr>
<tr>
<td>703</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuModuleUnload ( CUmodule hmod )</code><br>
Unloads a module.
</td>
</tr>
<tr>
<td>704</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name )</code><br>
Returns a function handle.
</td>
</tr>
<tr>
<td>705</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuModuleGetGlobal ( CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name )</code><br>
Returns a global pointer from a module.
</td>
</tr>
<tr>
<td>706</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuModuleGetTexRef ( CUtexref* pTexRef, CUmodule hmod, const char* name )</code><br>
Returns a handle to a texture reference.
</td>
</tr>
<tr>
<td>707</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuModuleGetSurfRef ( CUsurfref* pSurfRef, CUmodule hmod, const char* name )</code><br>
Returns a handle to a surface reference.
</td>
</tr>
<tr>
<td>708</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuLinkCreate ( unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut )</code><br>
Creates a pending JIT linker invocation.
</td>
</tr>
<tr>
<td>709</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuLinkAddData ( CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues )</code><br>
Add an input to a pending linker invocation.
</td>
</tr>
<tr>
<td>710</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuLinkAddFile ( CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues )</code><br>
Add a file input to a pending linker invocation.
</td>
</tr>
<tr>
<td>711</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuLinkComplete ( CUlinkState state, void** cubinOut, size_t* sizeOut )</code><br>
Complete a pending linker invocation.
</td>
</tr>
<tr>
<td>712</td>
<td>✗</td>
<td>✗</td>
</tr>

<tr>
<td colspan=3>
<code>CUresult cuLinkDestroy ( CUlinkState state )</code><br>
Destroys state for a JIT linker invocation.
</td>
</tr>
<tr>
<td>713</td>
<td>✗</td>
<td>✗</td>
</tr>
</table>
