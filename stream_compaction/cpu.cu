#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int sum = 0;
            for(int i = 0; i < n; i++){
                odata[i] = sum;
                sum += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int idx = 0;
            for(int i = 0; i < n; i++){
                if(idata[i] == 0) continue;
                odata[idx] = idata[i];
                idx++;
            }
            timer().endCpuTimer();
            return idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int *map = new int[n];
            int *scan = new int[n];
            int indice = 0;
            for(int i = 0; i < n; i++){
                scan[i] = indice;
                if(idata[i] != 0){
                    indice += 1;
                    map[i] = 1;
                }
                else {
                    map[i] = 0;
                }
            }
            for(int i = 0; i < n; i++){
                if(map[i] == 0) continue;
                odata[scan[i]] = idata[i];
            }
            timer().endCpuTimer();
            return scan[n-1] + map[n-1];
        }
    }
}
