#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.h>
#endif

#line 6

__kernel void
merge(const __global float* arr, __global float* res_arr,
      const unsigned int size, const unsigned int n)
{
    const unsigned int global_id = get_global_id(0);

    if (global_id < size)
    {
        const unsigned int begin1 = global_id / (n * 2) * n * 2;
        const unsigned int end1 = ((begin1 + n) >= size) ? size : (begin1 + n);
        const unsigned int len1 = end1 - begin1;
        const unsigned int begin2 = end1;
        const unsigned int end2 = ((begin2 + n) >= size) ? size : (begin2 + n);
        const unsigned int len2 = end2 - end1;

        const __global float* first_part = arr + begin1;
        const __global float* second_part = arr + begin2;

        __global float* res_arr_part = res_arr + begin1;

        // Есть от max(ind - len2, 0) до min(n, ind) меньших элементов в первом массиве.
        // Ищем кандидата в первой половине и во второй половине.

        const unsigned int i = global_id - begin1;
        unsigned int l = (i > len2) ? i - len2 : 0;
        unsigned int r = (n > i) ? i : n;

        while (l < r) {
            unsigned int mid = (l + r) / 2;
            if (first_part[mid] < second_part[i - mid - 1]) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }

        const unsigned int offset = r;

        const unsigned int i1 = offset;
        const unsigned int i2 = i - offset;

        if (i1 < len1)
            if (i2 >= len2 || first_part[i1] <= second_part[i2])
                res_arr_part[i] = first_part[i1];
            else
                res_arr_part[i] = second_part[i2];
        else
            res_arr_part[i] = second_part[i2];
    }
}
