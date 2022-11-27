void swap(__global float* f1, __global float* f2) {
    float tmp = *f1;
    *f1 = *f2;
    *f2 = tmp;
}

__kernel void
bitonic(__global float* as, unsigned int n, unsigned int partSize, unsigned int startPartSize) {
    const unsigned int id = get_global_id(0);

    const unsigned int startPartIndex = id / startPartSize;
    const unsigned int inPartIndex = id % partSize;
    const unsigned int halfPartSize = partSize / 2;

    if (inPartIndex < halfPartSize && id + halfPartSize < n) {
        if ((startPartIndex % 2 == 0 && as[id] > as[id + halfPartSize])
                || (startPartIndex % 2 == 1 && as[id] < as[id + halfPartSize])) {
            swap(&as[id], &as[id + halfPartSize]);
        }
    }
}
