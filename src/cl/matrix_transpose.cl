//----------------------------------------------------------------------------//
__kernel void
matrix_transpose_simple(__global const float* data,
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

//----------------------------------------------------------------------------//
__kernel void
matrix_transpose(__global const float* data,
                 __global       float* data_out,
                 unsigned int M,
                 unsigned int K)
{
    const unsigned int global_i = get_global_id(0);
    const unsigned int global_j = get_global_id(1);

    __local float working_part[WORKGROUP_SIZE][WORKGROUP_SIZE];

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    const unsigned int global_i_block_start = global_i - local_i;
    const unsigned int global_j_block_start = global_j - local_j;

    const unsigned int taken_i = global_i_block_start + local_i;
    const unsigned int taken_j = global_j_block_start + local_j;

    if (taken_j < M && taken_i < K) {
        // Ввод идёт последовательно по i.
        working_part[local_i][local_j] = data[taken_j * K + taken_i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int given_i = global_j_block_start + local_i;
    const unsigned int given_j = global_i_block_start + local_j;

    if (given_i < M && given_j < K) {
        // Вывод идёт последовательно по i.
        data_out[given_j * M + given_i] = working_part[local_j][local_i];
    }
}
