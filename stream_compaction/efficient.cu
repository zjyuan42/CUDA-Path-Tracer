#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        __global__ void Upsweep(int stride, int n, int *data){
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n) return;
            int left = (2 * idx + 1) * stride - 1;
            int right = (2 * idx + 2) * stride - 1;
            data[right] = data[left] + data[right];
            data[left] = data[left];
        }

        __global__ void Downsweep(int stride, int n, int *data){
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx >= n) return;
            int left = (2 * idx + 1) * stride - 1;
            int right = (2 * idx + 2) * stride - 1;
            int temp = data[left];
            data[left] = data[right];
            data[right] += temp;
        }

        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int *devdata;
            int padded = 1 << ilog2ceil(n);

            cudaMalloc(&devdata, sizeof(int) * padded);
            cudaMemset(devdata, 0, sizeof(int) * padded);
            cudaMemcpy(devdata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);


            int blockSize = 256;
            timer().startGpuTimer();
            for(int i = 1; i < padded; i *= 2){
                int gridNum = (padded / (i * 2) + blockSize - 1) / blockSize;
                Upsweep << <gridNum, blockSize>> > (i, padded / (i * 2), devdata);
            }
            cudaMemset(devdata + padded - 1, 0, sizeof(int));
            for(int i = padded / 2; i >= 1; i /= 2){
                int gridNum = (padded / (i * 2) + blockSize - 1) / blockSize;
                Downsweep << <gridNum, blockSize>> > (i, padded / (i * 2), devdata);
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, devdata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(devdata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        
        void scanDevice(int padded, int *devdata){
            int blockSize = 256;
            for(int i = 1; i < padded; i *= 2){
                int gridNum = (padded / (i * 2) + blockSize - 1) / blockSize;
                Upsweep << <gridNum, blockSize>> > (i, padded / (i * 2), devdata);
            }       
            cudaMemset(devdata + padded - 1, 0, sizeof(int));
            for(int i = padded / 2; i >= 1; i /= 2){
                int gridNum = (padded / (i * 2) + blockSize - 1) / blockSize;
                Downsweep << <gridNum, blockSize>> > (i, padded / (i * 2), devdata);
            }
        }
        
        int compact(int n, int *odata, const int *idata) {
            if(n < 1) return -1;
            int padded = 1 << ilog2ceil(n);
            int *map;
            int *indices;
            int *devidata;
            int *devodata;
            int blockSize = 256;
            int gridNum = (n + blockSize - 1) / blockSize;
            cudaMalloc(&map, sizeof(int) * n);
            cudaMalloc(&indices, sizeof(int) * padded);
            cudaMalloc(&devidata, sizeof(int) * n);
            cudaMalloc(&devodata, sizeof(int) * n);

            cudaMemset(map, 0, sizeof(int) * n);
            cudaMemset(indices, 0, sizeof(int) * padded);
            cudaMemcpy(devidata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);


            timer().startGpuTimer();
            // TODO
            StreamCompaction::Common::kernMapToBoolean<< <gridNum, blockSize>> > (n, map, devidata);
            cudaMemcpy(indices, map, sizeof(int) * n, cudaMemcpyDeviceToDevice);

            scanDevice(padded, indices);

            StreamCompaction::Common::kernScatter << <gridNum, blockSize>> > (n, devodata, devidata, map, indices);
            timer().endGpuTimer();
            int lastIndex = 0;
            int lastMap = 0;

            cudaMemcpy(
                &lastIndex,
                indices + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost);

            cudaMemcpy(
                &lastMap,
                map + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost);

            int ret = lastIndex + lastMap;
            cudaMemcpy(odata, devodata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(map);
            cudaFree(indices);
            cudaFree(devidata);
            cudaFree(devodata);

            return ret;
        }
    }
}
