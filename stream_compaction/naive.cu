#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernStep(int n, int *idata, int *odata, int offset){
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n) return;

            if(idx >= offset){
                odata[idx] = idata[idx - offset] + idata[idx];
            }
            else odata[idx] = idata[idx];
        }

        __global__ void kernInit(int n, int *idata, int *currentdata){
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n) return;

            currentdata[idx] = (idx == 0) ? 0 : idata[idx - 1];
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) return;
            // TODO
            int blockSize = 256;
            int Gridnum = (n + blockSize - 1) / blockSize;
            int *devidata;
            int *dev_idata;
            int *devodata;

            cudaMalloc(&devidata, sizeof(int) * n);
            cudaMemcpy(devidata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            cudaMalloc(&dev_idata, sizeof(int) * n);

            cudaMalloc(&devodata, sizeof(int) * n);
            timer().startGpuTimer();

            kernInit << <Gridnum, blockSize>> > (n, devidata, dev_idata);
            int turn = 1;
            while(turn <= n){
                if(turn > 1) std::swap(dev_idata, devodata);
                kernStep << <Gridnum, blockSize>> > (n, dev_idata, devodata, turn);
                turn *= 2;
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, devodata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(devidata);
            cudaFree(dev_idata);
            cudaFree(devodata);
        }
        
    }
}
