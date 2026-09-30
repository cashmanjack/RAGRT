#pragma once
#include <optix.h>
#include <cuda_runtime.h>

#define NUM_3D_SUBSPACES 32
#define NUM_GROUPS_PER_SUBSPACE 16
#define NUM_FINE_PER_SUBSPACE 256

struct CorrParams3D {
    float3* query_points;
    OptixTraversableHandle subspace_gas[NUM_3D_SUBSPACES];
    float3* centroid_xyz;
    int*    out_hit_centroid;
    float*  out_hit_value;
    int*    out_hit_count;
    float*  rt_table;
    int     max_hits;
    
    // Version B: Adaptive Hierarchical Pruning Parameters (256 Codewords)
    float3* super_centroid_xyz; // [32, 16]
    float*  group_radius;       // [32, 16]
    int*    group_assign;       // [32, 256]
    float   prune_threshold;
};
