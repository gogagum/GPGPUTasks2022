#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b,
               std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", "
                << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int)r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum,
                            "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd()
                  << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg()
                  << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum,
                            "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s"
                  << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg()
                  << " millions/s" << std::endl;
    }

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    unsigned int workGroupSize = 128;
    unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    {
        gpu::Context ctx;
        ctx.init(device.device_id_opencl);
        ctx.activate();

        gpu::gpu_mem_32u gpuNums;
        gpuNums.resizeN(n);
        gpuNums.writeN(as.data(), n);

        gpu::gpu_mem_32u gpuRes;
        gpuRes.resizeN(1);

        ocl::Kernel baseline_kernel(sum_kernel, sum_kernel_length, "sum_baseline");
        baseline_kernel.compile();

        timer t;
        for (std::size_t iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int res = 0;
            gpuRes.writeN(&res, 1);
            baseline_kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), gpuNums, gpuRes, n);
            gpuRes.readN(&res, 1);
            EXPECT_THE_SAME(reference_sum, res, "GPU baseline result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU baseline: " << t.lapAvg() << "+-" << t.lapStd() << " s"
                  << std::endl;
        std::cout << "GPU baseline: " << (n/1000.0/1000.0) / t.lapAvg()
                  << " millions/s" << std::endl;
    }

    {
      gpu::Context ctx;
      ctx.init(device.device_id_opencl);
      ctx.activate();

      gpu::gpu_mem_32u gpuNums;
      gpuNums.resizeN(n);
      gpuNums.writeN(as.data(), n);

      gpu::gpu_mem_32u gpuRes;
      gpuRes.resizeN(1);

      ocl::Kernel baseline_kernel(sum_kernel, sum_kernel_length, "sum_cycle");
      baseline_kernel.compile();

      timer t;
      for (std::size_t iter = 0; iter < benchmarkingIters; ++iter) {
        unsigned int res = 0;
        gpuRes.writeN(&res, 1);
        baseline_kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), gpuNums, gpuRes, n);
        gpuRes.readN(&res, 1);
        EXPECT_THE_SAME(reference_sum, res, "GPU cycle result should be consistent!");
        t.nextLap();
      }
      std::cout << "GPU cycle: " << t.lapAvg() << "+-" << t.lapStd() << " s"
                << std::endl;
      std::cout << "GPU cycle: " << (n/1000.0/1000.0) / t.lapAvg()
                << " millions/s" << std::endl;
    }

    {
      gpu::Context ctx;
      ctx.init(device.device_id_opencl);
      ctx.activate();

      gpu::gpu_mem_32u gpuNums;
      gpuNums.resizeN(n);
      gpuNums.writeN(as.data(), n);

      gpu::gpu_mem_32u gpuRes;
      gpuRes.resizeN(1);

      ocl::Kernel baseline_kernel(sum_kernel, sum_kernel_length, "sum_cycle_coalesced");
      baseline_kernel.compile();

      timer t;
      for (std::size_t iter = 0; iter < benchmarkingIters; ++iter) {
        unsigned int res = 0;
        gpuRes.writeN(&res, 1);
        baseline_kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), gpuNums, gpuRes, n);
        gpuRes.readN(&res, 1);
        EXPECT_THE_SAME(reference_sum, res, "GPU cycle_coalesced result should be consistent!");
        t.nextLap();
      }
      std::cout << "GPU cycle_coalesced: " << t.lapAvg() << "+-" << t.lapStd() << " s"
                << std::endl;
      std::cout << "GPU cycle_coalesced: " << (n/1000.0/1000.0) / t.lapAvg()
                << " millions/s" << std::endl;
    }

    {
      gpu::Context ctx;
      ctx.init(device.device_id_opencl);
      ctx.activate();

      gpu::gpu_mem_32u gpuNums;
      gpuNums.resizeN(n);
      gpuNums.writeN(as.data(), n);

      gpu::gpu_mem_32u gpuRes;
      gpuRes.resizeN(1);

      ocl::Kernel baseline_kernel(sum_kernel, sum_kernel_length, "sum_cycle_coalesced_local");
      baseline_kernel.compile();

      timer t;
      for (std::size_t iter = 0; iter < benchmarkingIters; ++iter) {
        unsigned int res = 0;
        gpuRes.writeN(&res, 1);
        baseline_kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), gpuNums, gpuRes, n);
        gpuRes.readN(&res, 1);
        EXPECT_THE_SAME(reference_sum, res, "GPU cycle_coalesced_local result should be consistent!");
        t.nextLap();
      }
      std::cout << "GPU cycle_coalesced_local: " << t.lapAvg() << "+-" << t.lapStd() << " s"
                << std::endl;
      std::cout << "GPU cycle_coalesced_local: " << (n/1000.0/1000.0) / t.lapAvg()
                << " millions/s" << std::endl;
    }

    {
      gpu::Context ctx;
      ctx.init(device.device_id_opencl);
      ctx.activate();

      gpu::gpu_mem_32u gpuNums;
      gpuNums.resizeN(n);
      gpuNums.writeN(as.data(), n);

      gpu::gpu_mem_32u gpuRes;
      gpuRes.resizeN(1);

      ocl::Kernel baseline_kernel(sum_kernel, sum_kernel_length, "sum_tree");
      baseline_kernel.compile();

      timer t;
      for (std::size_t iter = 0; iter < benchmarkingIters; ++iter) {
        unsigned int res = 0;
        gpuRes.writeN(&res, 1);
        baseline_kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), gpuNums, gpuRes, n);
        gpuRes.readN(&res, 1);
        EXPECT_THE_SAME(reference_sum, res, "GPU tree result should be consistent!");
        t.nextLap();
      }
      std::cout << "GPU tree: " << t.lapAvg() << "+-" << t.lapStd() << " s"
                << std::endl;
      std::cout << "GPU tree: " << (n/1000.0/1000.0) / t.lapAvg()
                << " millions/s" << std::endl;
    }
}
