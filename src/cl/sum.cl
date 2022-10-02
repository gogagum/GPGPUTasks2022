#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#ifndef VALUES_PER_WORK_ITEM
#define VALUES_PER_WORK_ITEM 64
#endif

#ifndef WORK_GROUP_SIZE
#define WORK_GROUP_SIZE 128
#endif

/*----------------------------------------------------------------------------*/
__kernel void sum_baseline(__global const unsigned int* array,
                           __global       unsigned int* ret,
                           unsigned int n)
{
    const size_t i = get_global_id(0);
    if (i < n)
    {
        atomic_add(ret, array[i]);
    }
}

/*----------------------------------------------------------------------------*/
__kernel void sum_cycle(__global const unsigned int* array,
                        __global       unsigned int* ret,
                        unsigned int n)
{
    const size_t global_id = get_global_id(0);
    if (global_id * VALUES_PER_WORK_ITEM >= n)
    {
        return;
    }
    const size_t finish = (global_id + 1) * VALUES_PER_WORK_ITEM > n
                  ? n % VALUES_PER_WORK_ITEM
                  : VALUES_PER_WORK_ITEM;
    unsigned int sum = 0;
    for (unsigned int i = 0; i < finish; ++i)
    {
        sum += array[global_id * VALUES_PER_WORK_ITEM + i];
    }
    atomic_add(ret, sum);
}

/*----------------------------------------------------------------------------*/
__kernel void sum_cycle_coalesced(__global const unsigned int* array,
                                  __global       unsigned int* ret,
                                  unsigned int n)
{
    const size_t local_id = get_local_id(0);
    const size_t group_id = get_group_id(0);
    const size_t group_size = get_local_size(0);
    unsigned int sum = 0;
    for (unsigned int i = 0; i < VALUES_PER_WORK_ITEM; ++i)
    {
        const size_t total_i =
            group_id * group_size * VALUES_PER_WORK_ITEM
            + i * group_size + local_id;
        if (total_i >= n)
        {
            break;
        }
        sum += array[total_i];
    }
    atomic_add(ret, sum);
}

/*----------------------------------------------------------------------------*/
__kernel void sum_cycle_coalesced_local(__global const unsigned int* array,
                                        __global       unsigned int* ret,
                                        unsigned int n)
{
    const size_t global_id = get_global_id(0);
    const size_t local_id = get_local_id(0);
    __local unsigned int local_array_copy[WORK_GROUP_SIZE];
    local_array_copy[local_id] = array[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0)
    {
        unsigned int local_sum = 0;
        for (unsigned int i = 0; i < WORK_GROUP_SIZE; ++i)
        {
            local_sum += local_array_copy[i];
        }
        atomic_add(ret, local_sum);
    }
}

/*----------------------------------------------------------------------------*/
__kernel void sum_tree(__global const unsigned int* array,
                       __global       unsigned int* ret,
                       unsigned int n)
{
    const size_t local_id = get_local_id(0);
    const size_t global_id = get_global_id(0);

    __local unsigned int local_array_copy[WORK_GROUP_SIZE];
    local_array_copy[local_id] = array[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int i = WORK_GROUP_SIZE; i > 1; i /= 2)
    {
        if (2 * local_id < i)
        {
            const unsigned int left = local_array_copy[local_id];
            const unsigned int right = local_array_copy[local_id+i/2];
            local_array_copy[local_id] = left + right;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0)
    {
        atomic_add(ret, local_array_copy[0]);
    }
}
