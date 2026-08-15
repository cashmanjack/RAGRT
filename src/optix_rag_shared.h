#pragma once
#include <optix.h>
#include <vector_types.h>
#include <cstdint>

// ---------------------------------------------------------------------
// Test constants -- keep validate_optix_rag.py in sync with these
// ---------------------------------------------------------------------
constexpr int M             = 2;                  // 2D subspaces
constexpr int NUM_SUBSPACES = 2;                   // D = 4 in this test
constexpr int NUM_CENTROIDS = 5;                   // 5 centroids per subspace
constexpr int NUM_SPHERES   = NUM_CENTROIDS * NUM_SUBSPACES;
constexpr int MAX_HITS      = NUM_CENTROIDS;       // one ray only hits its own subspace
constexpr float RADIUS      = 1.0f;                // L2 threshold for now

struct SphereSpec {
    float x, y;
    int subspace_id;
};

// Per-ray scratch data. Passed by POINTER via two payload registers
// (see packPointer/unpackPointer in optix_rag_device.cu) rather than
// spread across 7 individual payload registers -- growable, and avoids
// hand-packing an array across fixed OptiX payload slots.
struct RayPayload {
    int   subspace_id;
    int   hit_count;
    float hit_ts[MAX_HITS];
};

struct HitSbtData {
    float3 center;
    float  radius;
    int    subspace_id;
};

struct LaunchParams {
    float2* query_subspaces;
    float*  distances;                 // NUM_SUBSPACES * MAX_HITS floats
    int*    counts;                    // NUM_SUBSPACES ints
    OptixTraversableHandle gas_handle;
};
