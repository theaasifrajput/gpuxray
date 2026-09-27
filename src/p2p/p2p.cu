#include "p2p/p2p.h"
#include "common/cuda_check.h"

#include <cuda_runtime.h>

#include <vector>

std::vector<P2PLink> discover_p2p_links() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::vector<P2PLink> links;

    if (device_count < 2) {
        return links;
    }

    links.reserve(
        static_cast<std::size_t>(device_count) *
        static_cast<std::size_t>(device_count - 1)
    );

    for (int source = 0; source < device_count; ++source) {
        for (int destination = 0;
             destination < device_count;
             ++destination) {

            if (source == destination) {
                continue;
            }

            int can_access = 0;

            CUDA_CHECK(cudaDeviceCanAccessPeer(
                &can_access,
                source,
                destination
            ));

            links.push_back({
                source,
                destination,
                can_access != 0
            });
        }
    }

    return links;
}