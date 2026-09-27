#pragma once

#include <vector>

struct P2PLink {
    int source;
    int destination;
    bool supported;
};

std::vector<P2PLink> discover_p2p_links();