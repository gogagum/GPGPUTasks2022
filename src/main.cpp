#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <array>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <vector>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init()) {
        throw std::runtime_error("Can't init OpenCL driver!");
    }

    // 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    struct DeviceInfo {
        cl_platform_id pl_id;
        cl_device_id id;
        bool gpu;
        cl_ulong mem;

        // Компаратор, который выдаёт true если первое устройство лучше второго.
        struct SuitabilityCmp {
            bool operator()(const DeviceInfo& f, const DeviceInfo& s) {
                if (f.gpu && !s.gpu) {
                    return true;
                }
                if (!f.gpu && s.gpu) {
                    return false;
                }
                return f.mem > s.mem;
            }
        };
    };


    std::priority_queue<DeviceInfo, std::vector<DeviceInfo>, DeviceInfo::SuitabilityCmp> devicesCompared;

    for (auto platform: platforms) {
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_AVAILABLE, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_AVAILABLE, devicesCount, devices.data(), nullptr));
        for (auto device: devices) {
            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
            auto isGpu = static_cast<bool>(deviceType & CL_DEVICE_TYPE_GPU);
            cl_ulong deviceMaxMemoryAllocCapacity;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                          sizeof(deviceMaxMemoryAllocCapacity), &deviceMaxMemoryAllocCapacity,
                                          nullptr));
            devicesCompared.push({platform, device, isGpu, deviceMaxMemoryAllocCapacity});
        }
    }

    auto bestDevice = devicesCompared.top();

    if (bestDevice.gpu) {
        std::cout << "Found GPU!" << std::endl;
    } else {
        std::cout << "No GPU. Using other type of device." << std::endl;
    }


    // 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime ->
    // Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод
    // возвращает код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)

    cl_context_properties* propsPtr = nullptr;
    std::array<cl_device_id, 1> devicesList = {bestDevice.id};
    cl_int errcode_ret;

    cl_context ctx = clCreateContext(propsPtr, devicesList.size(), devicesList.data(), nullptr, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    // 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime ->
    // Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)

    cl_command_queue cmdQueue = clCreateCommandQueue(ctx, bestDevice.id, 0, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    unsigned int n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух
    // суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n
    // штук.
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data()
    // (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL
    // под-программы, кернела и т.п. тоже нужно освобождать)

    cl_mem buffA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * n, as.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    cl_mem buffB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * n, bs.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    cl_mem buffC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * n, cs.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    // 6 Выполните 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
    }

    // 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель

    const char* strPtr = kernel_sources.c_str();
    cl_program clProgram = clCreateProgramWithSource(ctx, 1, &strPtr, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    // 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram

    const cl_int buildRes = clBuildProgram(clProgram, devicesList.size(), devicesList.data(), nullptr, nullptr, nullptr);

    std::size_t logSize = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(clProgram, devicesList[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize));

    if (logSize > 1) {
        std::string logStr(logSize, 0);
        OCL_SAFE_CALL(clGetProgramBuildInfo(clProgram, devicesList[0], CL_PROGRAM_BUILD_LOG, logSize,
                                            logStr.data(), nullptr));

        std::cout << "Log:" << std::endl;
        std::cout << logStr << std::endl;
    }
    OCL_SAFE_CALL(buildRes);

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е.
    // когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается,
    // какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    //    size_t log_size = 0;
    //    std::vector<char> log(log_size, 0);
    //    if (log_size > 1) {
    //        std::cout << "Log:" << std::endl;
    //        std::cout << log.data() << std::endl;
    //    }

    // 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects

    cl_kernel kernel = clCreateKernel(clProgram, "aplusb", &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    // Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь,
    // что тип количества элементов такой же в кернеле)
    {
        OCL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffA));
        OCL_SAFE_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffB));
        OCL_SAFE_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffC));
        OCL_SAFE_CALL(clSetKernelArg(kernel, 3, sizeof(unsigned int), &n));
    }

    // 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        const std::size_t workGroupSize = 128;
        const std::size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(cmdQueue, kernel, 1, nullptr, &global_work_size,
                                                 &workGroupSize, 0, nullptr, &event));
            clWaitForEvents(1, &event);
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / t.lapAvg() * 1e-9  << std::endl;

        // 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3.*n*sizeof(float) / (t.lapAvg() * std::pow(1024, 3)) << " GB/s" << std::endl;
    }

    // 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и
    // рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(cmdQueue, buffC, CL_TRUE, 0, sizeof(float) * n, cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << n*sizeof(float) / (t.lapAvg() * std::pow(1024, 3)) << " GB/s" << std::endl;
    }

    // 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать
    // намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseMemObject(buffA));
    OCL_SAFE_CALL(clReleaseMemObject(buffB));
    OCL_SAFE_CALL(clReleaseMemObject(buffC));
    OCL_SAFE_CALL(clReleaseCommandQueue(cmdQueue));
    OCL_SAFE_CALL(clReleaseContext(ctx));

    return 0;
}
