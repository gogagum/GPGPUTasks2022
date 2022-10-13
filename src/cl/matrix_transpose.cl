#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.h>
#endif

#line 6

#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE 16
#endif

__kernel void matrix_transpose_simple(__global const float* data,
                                      __global       float* data_out,
                                      unsigned int M,
                                      unsigned int K)
{
  unsigned int global_i = get_global_id(0);
  unsigned int global_j = get_global_id(1);

  if (global_i < M && global_j < K) {
    unsigned int local_i = get_local_id(0);
    unsigned int local_j = get_local_id(1);
    data_out[global_j * M + global_i] = data[global_i * K + global_j];
  }
}

__kernel void matrix_transpose(__global const float* data,
                               __global       float* data_out,
                               unsigned int M,
                               unsigned int K)
{
    unsigned int global_i = get_global_id(0);
    unsigned int global_j = get_global_id(1);

    __local float working_part[WORKGROUP_SIZE][WORKGROUP_SIZE];

    if (global_i < M && global_j < K) {
      unsigned int local_i = get_local_id(0);
      unsigned int local_j = get_local_id(1);
      working_part[local_i][local_j] = data[global_i * K + global_j];
      data_out[global_j * M + global_i] = working_part[local_i][local_j];
    }
}
