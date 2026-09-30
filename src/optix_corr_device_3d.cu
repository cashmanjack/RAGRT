#include <optix.h>
#include <math.h>
#include "optix_corr_shared_3d.h"

extern "C" __constant__ CorrParams3D params;

extern "C" __global__ void __raygen__corr_3d() {
    uint3 idx = optixGetLaunchIndex();
    int total_rays = optixGetLaunchDimensions().x;
    int nq = total_rays / NUM_3D_SUBSPACES;

    int subspace = idx.x / nq;
    int query_id = idx.x % nq;
    int logical_ray_id = query_id * NUM_3D_SUBSPACES + subspace;

    float3 q = params.query_points[logical_ray_id];
    float norm = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);
    if (norm < 1e-6f) norm = 1.0f;

    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 dir    = make_float3(q.x / norm, q.y / norm, q.z / norm);

    unsigned int p0 = (unsigned int)logical_ray_id;
    OptixTraversableHandle gas = params.subspace_gas[subspace];

    optixTrace(gas, origin, dir, 0.0f, 2.0f, 0.0f,
               255, OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0);
}

extern "C" __global__ void __anyhit__corr_3d() {
    int logical_ray_id = (int)optixGetPayload_0();
    int subspace = logical_ray_id % NUM_3D_SUBSPACES;

    int prim_idx = optixGetPrimitiveIndex();
    int entry_id = (prim_idx / 4) % NUM_FINE_PER_SUBSPACE; // 8-bit (0 .. 255)

    float3 q = params.query_points[logical_ray_id];
    float q_norm = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);

    // Version B: Adaptive Subspace Energy Pruning in OptiX Silicon
    if (params.prune_threshold > -100.0f && params.group_assign != nullptr) {
        int group_id = params.group_assign[subspace * NUM_FINE_PER_SUBSPACE + entry_id];
        float3 super_c = params.super_centroid_xyz[subspace * NUM_GROUPS_PER_SUBSPACE + group_id];
        float r = params.group_radius[subspace * NUM_GROUPS_PER_SUBSPACE + group_id];

        // Adaptive Cauchy-Schwarz bound check scaled by subspace signal energy ||q_s||
        float max_bound = (q.x * super_c.x + q.y * super_c.y + q.z * super_c.z) + (q_norm * r);
        if (max_bound < params.prune_threshold * q_norm) {
            optixIgnoreIntersection();
            return;
        }
    }

    float3 c = params.centroid_xyz[subspace * NUM_FINE_PER_SUBSPACE + entry_id];
    float value = q.x * c.x + q.y * c.y + q.z * c.z;

    int cnt = atomicAdd(&params.out_hit_count[logical_ray_id], 1);
    if (cnt < params.max_hits) {
        int base = logical_ray_id * params.max_hits + cnt;
        params.out_hit_centroid[base] = entry_id;
        params.out_hit_value[base]    = value;
    }

    if (params.rt_table != nullptr) {
        params.rt_table[logical_ray_id * NUM_FINE_PER_SUBSPACE + entry_id] = value;
    }

    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__corr_3d() {}
