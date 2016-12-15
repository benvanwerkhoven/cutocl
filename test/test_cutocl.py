from cutocl import cutocl

def test_replace_builtins():
    input_string = " threadIdx.z blockDim.x gridDim.x blockIdx.y "
    reference = " get_local_id(2) get_group_size(0) get_num_groups(0) get_group_id(1) "
    answer = cutocl._replace_builtins(input_string)
    print(answer)
    assert answer == reference

def test_replace_qualifiers():
    input_string = " __global__ __shared__ __constant__ __device__ __host__ __forceinline__ "
    reference = " __kernel __local __constant   inline "
    answer = cutocl._replace_qualifiers(input_string)
    print(answer)
    assert answer == reference

def test_fix_parameter_list1():
    input_string = " __kernel void blabla(float * a, int b, char * c, const double *__restrict__d);"
    reference = " __kernel void blabla(__global float * a, int b, __global char * c, __global const double *__restrict__d);"
    answer = cutocl._fix_parameter_list(input_string)
    print(answer)
    assert answer == reference

def test_translate():
    kernel_string = """ __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """
    reference = """ __kernel void vector_add(__global float *c, __global float *a, __global float *b, int n) {
        int i = get_group_id(0) * block_size_x + get_local_id(0);
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """
    answer = cutocl.translate(kernel_string)
    print("[" + answer + "]")
    assert answer == reference
