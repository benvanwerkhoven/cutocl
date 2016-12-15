#!/usr/bin/env python
import sys
import os
import re

def _replace_builtins(input_string):
    """ replaces CUDA built-in variables with OpenCL built-in function equivalents """
    def repl(x):
        builtins = {'threadIdx':'get_local_id', 'blockIdx':'get_group_id',
                    'blockDim':'get_group_size', 'gridDim': 'get_num_groups'}
        dims = {'x':'0', 'y':'1', 'z':'2'}
        return builtins[x.group(1)] + '(' + dims[x.group(2)] + ')'
    return re.sub(r'(threadIdx|blockIdx|blockDim|gridDim)\.(x|y|z)', repl, input_string)

def _replace_qualifiers(input_string):
    replace = { '__global__':'__kernel', '__shared__':'__local', '__constant__':'__constant',
                '__device__':'', '__host__':'', '__syncthreads()':'barrier(CLK_LOCAL_MEM_FENCE)',
                '__forceinline__':'inline'}
    for k,v in replace.items():
        input_string = input_string.replace(k,v)
    return input_string

def _fix_parameter_list(input_string):
    """ inserts __global in front of all pointer arguments in the parameter list """
    def repl(x):
        return x.group(0).replace(x.group(1), re.sub(r'(\s*)([^,\*]*\*[^,]*)(,|$)', r'\1__global \2\3', x.group(1)))
    return re.sub(r'__kernel .*?\((.*?),*\)\s*[{|;]', repl, input_string, flags=re.DOTALL)

def translate(input_string):
    """ convert a CUDA kernels to OpenCL kernels

    :param input_string: The string containing the CUDA code.
    :type input_string: string

    :returns: A string containing the roughly translated code for OpenCL kernels
    :rtype: string
    """
    input_string = _replace_builtins(input_string)
    input_string = _replace_qualifiers(input_string)
    return _fix_parameter_list(input_string)

def main():
    if len(sys.argv) != 2:
        print("Usage:\n cutocl <filename>")
        exit(1)
    if not os.path.exists(sys.argv[1]):
        print("cutocl error: input file does not exist")
        exit(1)
    with open(sys.argv[1], 'r') as f:
        input_string = f.read()
    print(translate(input_string))

if __name__ == "__main__":
    main()

