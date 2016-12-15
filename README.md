CUDA to OpenCL
===============================
A Python tool for quick and dirty translation of CUDA kernels to OpenCL

Documentation
-------------
You could include a link to the full documentation of your project here.

Example Usage
-------------
As a command-line tool:  
```bash
cutocl vector_add.cu
```

Or use from Python:  
```python
from cutocl import cutocl
kernel_string = """
__global__ void vector_add(float *c, float *a, float *b, int n) {
    int i = blockIdx.x * block_size_x + threadIdx.x;
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
"""
opencl_kernel = cutocl.translate(kernel_string)
print(opencl_kernel)
```
This should produce:  
```opencl
__kernel void vector_add(__global float *c, __global float *a, __global float *b, int n) {
    int i = get_group_id(0) * block_size_x + get_local_id(0);
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
```

Installation
------------
clone the repository  
    `git clone git@github.com:benvanwerkhoven/cutocl.git`  
change into the top-level directory  
    `cd cutocl`  
install using  
    `pip install .`

Dependencies
------------
 * Python 3

License
-------
Copyright (c) 2016, Ben van Werkhoven

Apache Software License 2.0

Contributing
------------
Contributing authors so far:
* Ben van Werkhoven


