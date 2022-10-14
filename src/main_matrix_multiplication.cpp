#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int M = 1024;
    unsigned int K = 1024;
    unsigned int N = 1024;
    const double gflops =
        (static_cast<double>(M) * K * N * 2) / (1000 * 1000 * 1000);
    // Умножить на два, т.к. операция сложения и умножения.

    std::vector<float> as(M*K, 0);
    std::vector<float> bs(K*N, 0);
    std::vector<float> cs(M*N, 0);

    FastRandom r(M+K+N);
    for (float & a : as) {
        a = r.nextf();
    }
    for (float & b : bs) {
        b = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N
              << "!" << std::endl;

    as = {1.f, 2.f, 3.f, 5.f};
    bs = {7.f, 11.f, 13.f, 17.f};

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as[j * K + k] * bs[k * N + i];
                    }
                    cs[j * N + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    const std::vector<float> cs_cpu_reference = cs;

    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    cs_gpu.resizeN(M*N);

    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);

    const std::size_t block_size = 16;
    const std::string def = " -D BLOCK_SIZE=" + std::to_string(block_size);

    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication,
                                             matrix_multiplication_length,
                                             "matrix_multiplication",
                                             def);
    matrix_multiplication_kernel.compile();

    {
        timer t;
        const std::size_t piece_size = 32;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            const unsigned int squares_N = (N - 1) / piece_size + 1;
            const unsigned int squares_M = (M - 1) / piece_size + 1;
            const unsigned int global_work_size_N = squares_N * block_size;
            const unsigned int global_work_size_M = squares_M * block_size;
            matrix_multiplication_kernel.exec(gpu::WorkSize(block_size,
                                                            block_size,
                                                            global_work_size_N,
                                                            global_work_size_M),
                                              as_gpu, bs_gpu, cs_gpu, M, K, N);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs_gpu.readN(cs.data(), M*N);

    // Проверяем корректность результатов
    double diff_sum = 0;
    for (int i = 0; i < M * N; ++i) {
        double a = cs[i];
        double b = cs_cpu_reference[i];
        if (a != 0.0 && b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            diff_sum += diff;
        }
    }

    double diff_avg = diff_sum / (M * N);
    std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01 || diff_avg != diff_avg /*NaN*/) {
        std::cerr << "Too big difference!" << std::endl;
        return 1;
    }

    return 0;
}
