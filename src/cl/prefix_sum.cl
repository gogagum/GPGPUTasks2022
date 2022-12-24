//----------------------------------------------------------------------------//
__kernel void
reduce2(__global const unsigned int* currReduced,
        __global       unsigned int* res,
        unsigned int n) {
    const unsigned int id = get_global_id(0);

    if (id * 2 + 1 < n) {
        res[id] = currReduced[id * 2] + currReduced[id * 2 + 1];
    } else if (id * 2 < n) {
        res[id] = currReduced[id * 2];
    } else if (id < n) {
        res[id] = 42;
    }
}

//----------------------------------------------------------------------------//
__kernel void
prefix_sum_impl(__global const unsigned int* currReduced,
                __global       unsigned int* ret,
                unsigned int stage,
                unsigned int size) {
    const unsigned int id = get_global_id(0);
    const unsigned int i = (id + 1) >> stage;

    if (id <= size) {
        if (i & 1 == 1) {
            ret[id] += currReduced[i-1];
        }
    }
}
