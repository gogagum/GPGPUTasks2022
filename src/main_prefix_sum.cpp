#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <iostream>
#include <limits>
#include <vector>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"
#include "libgpu/work_size.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
	if (a != b) {
		std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
		throw std::runtime_error(message);
	}
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

	int benchmarkingIters = 10;
    unsigned int max_n = (1 << 20);

    for (unsigned int n = 2; n <= max_n; n *= 2) {
		std::cout << "______________________________________________" << std::endl;
		unsigned int values_range = std::min<unsigned int>(1023, std::numeric_limits<int>::max() / n);
		std::cout << "n=" << n << " values in range: [" << 0 << "; " << values_range << "]" << std::endl;

        //n += 3;  // Проверка, что случай не 2^n работает ОК.

		std::vector<unsigned int> as(n, 0);
		FastRandom r(n);
		for (int i = 0; i < n; ++i) {
			as[i] = r.next(0, values_range);
		}

		std::vector<unsigned int> bs(n, 0);
		{
			for (int i = 0; i < n; ++i) {
				bs[i] = as[i];
				if (i) {
					bs[i] += bs[i-1];
				}
			}
		}
		const std::vector<unsigned int> reference_result = bs;

		{
			{
				std::vector<unsigned int> result(n);
				for (int i = 0; i < n; ++i) {
					result[i] = as[i];
					if (i) {
						result[i] += result[i-1];
					}
				}
				for (int i = 0; i < n; ++i) {
					EXPECT_THE_SAME(reference_result[i], result[i], "CPU result should be consistent!");
				}
			}

			std::vector<unsigned int> result(n);
			timer t;
			for (int iter = 0; iter < benchmarkingIters; ++iter) {
				for (int i = 0; i < n; ++i) {
					result[i] = as[i];
					if (i) {
						result[i] += result[i-1];
					}
				}
				t.nextLap();
			}
			std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
			std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
		}

        std::vector<unsigned int> gpuResult(n, 0);

        {
            gpu::gpu_mem_32u prefixSumsGPU;
            prefixSumsGPU.resizeN(n);
            prefixSumsGPU.writeN(as.data(), n);
            gpu::gpu_mem_32u currReduced;
            currReduced.resizeN(n);
            gpu::gpu_mem_32u tmpReduced;
            tmpReduced.resizeN(n);

            ocl::Kernel reduce2(prefix_sum_kernel,
                                prefix_sum_kernel_length,
                                "reduce2");
            reduce2.compile();

            ocl::Kernel prefixSumImpl(prefix_sum_kernel,
                                      prefix_sum_kernel_length,
                                      "prefix_sum_impl");
            prefixSumImpl.compile();

            const auto calcWorkSize = [](unsigned int sz) {
                const unsigned int workGroupSize = 128;
                auto globalSize = (sz + workGroupSize - 1) / workGroupSize * workGroupSize;
                return gpu::WorkSize(workGroupSize, globalSize);
            };


            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                currReduced.writeN(as.data(), n);
                prefixSumsGPU.writeN(std::vector<unsigned int>(n, 0).data(), n);

                for (unsigned int stage = 0; ((n + 1) >> stage) > 0; ++stage) {
                    reduce2.exec(calcWorkSize((n + 1) >> stage), currReduced, tmpReduced, n);

                    prefixSumImpl.exec(calcWorkSize(n), currReduced, prefixSumsGPU,
                                       stage, n);
                    currReduced.swap(tmpReduced);
                }

                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

            prefixSumsGPU.readN(gpuResult.data(), n);
		}

        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(bs[i], gpuResult[i], "GPU results should be equal to CPU results!");
        }

	}
}
