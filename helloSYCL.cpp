// STD
#include <vector>
#include <stddef.h>
#include <iostream>

// SYCL header file
#include <CL/sycl.hpp>

// SYCL MKL header file
#include "mkl_sycl.hpp"

int main()
{

    const size_t M(3);      // Rows of our array
    const size_t N(3);      // Cols of our array
    const size_t sz(M * N); // Total elements

    std::vector<float> hostVec(sz); // To store result of GEMM

    {
        // Get information about all devices present on the system
        std::vector<cl::sycl::device> allDevices = cl::sycl::device::get_devices();

        // We pick first available device
        cl::sycl::device myDevice = allDevices[1];

        // Print out device name
        for (auto i = 0; i < allDevices.size(); i++)
            std::cout << "Device Name: " << i << " " << allDevices[i].get_info<cl::sycl::info::device::name>() << std::endl;

        // We can only submit jobs in a queue on a device
        cl::sycl::queue myQueue(myDevice);

        // We reserve a buffer with uninitialized memory
        auto size = cl::sycl::range<1>{sz}; // Interpreting sizes in a way that SYCL understands
        cl::sycl::buffer<float> buffA(size);

        // Then we fill it using a (templated) kernel
        myQueue.submit([&](cl::sycl::handler &h) {
            // We can only access buffers through "accessors"
            auto accDev = buffA.get_access<cl::sycl::access::mode::discard_write>(h);

            // Kernel invoked as a named lambda
            h.parallel_for<struct myKernel>(
                size, [=](cl::sycl::id<1> myID) {
                    accDev[myID] = 5.0f;
                });
        });

        // We reserve a buffer with uninitialized memory for storing result
        cl::sycl::buffer<float> buffB(size);

        // Call MKL GEMM (A*A = B)
        mkl::blas::gemm(myQueue, mkl::transpose::N, mkl::transpose::N, M, M, M, 1.0f, buffA, M, buffA, M, 0, buffB, M);

        // Copy back into the host Array
        // BUT SYCL only deals in buffers!! So lets wrap the host Array with a buffer and explicitly prevent copying by telling it to use the host pointer directly
        auto myTempBuffer = cl::sycl::buffer<float>(
            hostVec.data(), size,
            cl::sycl::property::buffer::use_host_ptr());

        myQueue.submit([&](cl::sycl::handler &h) {
            auto dest = myTempBuffer.get_access<cl::sycl::access::mode::discard_write>(h);
            auto src = buffB.get_access<cl::sycl::access::mode::read>(h);
            h.copy(src, dest);
        });

        // Sync all jobs before proceeding to exit program
        myQueue.wait();

        // When myTempBuffer goes out of scope, hostVec is updated to reflect the state in myTempBuffer
    }

    return 0;
}