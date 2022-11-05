#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.h>
#endif

#line 6

__kernel void matrix_transpose_simple(__global const float* data,
                                      __global       float* data_out,
                                      unsigned int M,
                                      unsigned int K)
{
  const unsigned int global_i = get_global_id(0);
  const unsigned int global_j = get_global_id(1);

  if (global_i < M && global_j < K) {
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);
    data_out[global_j * M + global_i] = data[global_i * K + global_j];
  }
}

__kernel void matrix_transpose(__global const float* data,
                               __global       float* data_out,
                               unsigned int M,
                               unsigned int K)
{
    const unsigned int global_i = get_global_id(0);
    const unsigned int global_j = get_global_id(1);

    __local float working_part[WORKGROUP_SIZE][WORKGROUP_SIZE];

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    if (global_i < M && global_j < K) {
      working_part[local_i][local_j] = data[global_j * K + global_i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_j < K && global_i < M) {
        data_out[global_i * M + global_j] = working_part[local_i][local_j];
    }
}
