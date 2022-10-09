#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b,
               const std::string& message,
               const std::string& filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", "
                << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

const unsigned int workGroupSize = 128;

void perform_gpu_calculations(const gpu::Device& device,
                              const std::vector<unsigned int>& as,
                              const std::string& kernelName,
                              std::size_t iterationsCnt,
                              unsigned int globalWorkSize,
                              unsigned int referenceSum) {
  gpu::Context ctx;
  ctx.init(device.device_id_opencl);
  ctx.activate();

  gpu::gpu_mem_32u gpuNums;
  gpuNums.resizeN(as.size());
  gpuNums.writeN(as.data(), as.size());

  gpu::gpu_mem_32u gpuRes;
  gpuRes.resizeN(1);

  ocl::Kernel baseline_kernel(sum_kernel, sum_kernel_length, kernelName);
  baseline_kernel.compile();

  timer t;
  for (std::size_t iter = 0; iter < iterationsCnt; ++iter) {
      unsigned int res = 0;
      gpuRes.writeN(&res, 1);
      baseline_kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                           gpuNums, gpuRes, static_cast<unsigned int>(as.size()));
      gpuRes.readN(&res, 1);
      EXPECT_THE_SAME(referenceSum, res,
                      "GPU " + kernelName + " result should be consistent!");
      t.nextLap();
  }
  std::cout << "GPU " + kernelName + ": " << t.lapAvg() << "+-" << t.lapStd()
            << " s" << std::endl;
  std::cout << "GPU " + kernelName + ": "
            << (double(as.size())/1000.0/1000.0) / t.lapAvg()
            << " millions/s" << std::endl;

}

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = static_cast<unsigned int>(r.next(0, std::numeric_limits<int>::max() / int(n)));
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

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        unsigned int globalWorkSize =
            (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        const size_t gpuBenchmarkIters = 50;

        perform_gpu_calculations(device, as, "sum_baseline", gpuBenchmarkIters,
                                 globalWorkSize, reference_sum);

        perform_gpu_calculations(device, as, "sum_cycle", gpuBenchmarkIters,
                                 globalWorkSize, reference_sum);

        unsigned int valuesPerWorkItem = 64;

        perform_gpu_calculations(device, as, "sum_cycle_coalesced",
                                 gpuBenchmarkIters, globalWorkSize / valuesPerWorkItem,
                                 reference_sum);

        perform_gpu_calculations(device, as, "sum_cycle_coalesced_local",
                                 gpuBenchmarkIters, globalWorkSize,
                                 reference_sum);

        perform_gpu_calculations(device, as, "sum_tree", gpuBenchmarkIters,
                                 globalWorkSize, reference_sum);
    }
}
