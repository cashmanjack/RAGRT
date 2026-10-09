#pragma once
#include <optix.h>
#include <cuda_runtime.h>

#define NUM_3D_SUBSPACES 32
#define NUM_FINE_PER_SUBSPACE 256

struct CorrParams3D {
    float3* query_points;
    OptixTraversableHandle subspace_gas[NUM_3D_SUBSPACES];
    float3* centroid_xyz;
    int*    out_hit_centroid;
    float*  out_hit_value;
    int*    out_hit_count;
    int     max_hits;
};
