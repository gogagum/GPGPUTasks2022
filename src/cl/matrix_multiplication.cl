#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/cliob_defines.h>
#endif

#line 6

__kernel void matrix_multiplication(__global const float* first_data,
                                    __global const float* second_data,
                                    __global       float* result_data,
                                    unsigned int N,
                                    unsigned int M,
                                    unsigned int K)
{
  __local float first_local_block[BLOCK_SIZE][BLOCK_SIZE];
  __local float second_local_block[BLOCK_SIZE][BLOCK_SIZE];

  // i: 1..N -- индекс строки в первой матрице
  // j: 1..M -- индекс столбца в первой матрице и строки во второй
  // l: 1..K -- индекс столбца во второй матрице

  const unsigned int i_local = get_local_id(0);  // -- индекс строки внутри блока
  const unsigned int l_local = get_local_id(1);  // -- индекс столбца внутри блока
  const unsigned int i = get_group_id(0) * BLOCK_SIZE + i_local;
  const unsigned int l = get_group_id(1) * BLOCK_SIZE + l_local;

  float result = 0;
  for (unsigned int j_base = 0; j_base < M; j_base += BLOCK_SIZE) {
    // Фиксируем вторую координату в первой матрице и первую во второй на уровне
    // индекса блока -- k_base.
    const unsigned int j_a = j_base + l_local;  // Индекс столбца в первой матрице
    const unsigned int j_b = j_base + i_local;  // Индекс строки во второй матрице
    first_local_block[i_local][l_local] = (i < N && j_a < M)
                                          ? first_data[i * N + j_a]
                                          : 0;
    second_local_block[i_local][l_local] =
                                          (l < K && j_b < M)
                                          ? second_data[j_b * K + l]
                                          : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int k_local = 0; k_local < BLOCK_SIZE; ++k_local) {
      result += first_local_block[i_local][k_local]
                * second_local_block[k_local][l_local];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (i < N && l < K) {
    result_data[i * M + l] = result;
  }
}
