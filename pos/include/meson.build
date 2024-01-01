# close all default macros, which is enable for editing
add_global_arguments('-DPOS_ENABLE_MACRO_FOR_DEBUG', language: ['c', 'cpp'])

if enable_runtime_debug_check == 1
    add_global_arguments('-DPOS_ENABLE_DEBUG_CHECK=1', language: ['c', 'cpp'])
else
    add_global_arguments('-DPOS_ENABLE_DEBUG_CHECK=0', language: ['c', 'cpp'])
endif

if enable_hijack_api_check == 1
    add_global_arguments('-DPOS_ENABLE_HIJACK_API_CHECK=1', language: ['c', 'cpp'])
else
    add_global_arguments('-DPOS_ENABLE_HIJACK_API_CHECK=0', language: ['c', 'cpp'])
endif

if enable_print_error == 1
    add_global_arguments('-DPOS_PRINT_ERROR=1', language: ['c', 'cpp'])
else
    add_global_arguments('-DPOS_PRINT_ERROR=0', language: ['c', 'cpp'])
endif

if enable_print_warn == 1
    add_global_arguments('-DPOS_PRINT_WARN=1', language: ['c', 'cpp'])
else
    add_global_arguments('-DPOS_PRINT_WARN=0', language: ['c', 'cpp'])
endif

if enable_print_log == 1
    add_global_arguments('-DPOS_PRINT_LOG=1', language: ['c', 'cpp'])
else
    add_global_arguments('-DPOS_PRINT_LOG=0', language: ['c', 'cpp'])
endif

if enable_print_debug == 1
    add_global_arguments('-DPOS_PRINT_DEBUG=1', language: ['c', 'cpp'])
else
    add_global_arguments('-DPOS_PRINT_DEBUG=0', language: ['c', 'cpp'])
endif

if enable_print_with_color == 1
    add_global_arguments('-DPOS_PRINT_WITH_COLOR=1', language: ['c', 'cpp'])
else
    add_global_arguments('-DPOS_PRINT_WITH_COLOR=0', language: ['c', 'cpp'])
endif

