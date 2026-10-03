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
    
    // Fast SFU reciprocal square root
    float dot_q = q.x * q.x + q.y * q.y + q.z * q.z;
    float inv_norm = rsqrtf(dot_q + 1e-12f);

    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 dir    = make_float3(q.x * inv_norm, q.y * inv_norm, q.z * inv_norm);

    unsigned int p0 = (unsigned int)logical_ray_id;
    unsigned int p1 = 0;
    OptixTraversableHandle gas = params.subspace_gas[subspace];

    optixTrace(gas, origin, dir, 0.0f, 2.0f, 0.0f,
               255, OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1);

    params.out_hit_count[logical_ray_id] = (int)p1;
}

extern "C" __global__ void __anyhit__corr_3d() {
    int logical_ray_id = (int)optixGetPayload_0();
    int subspace = logical_ray_id % NUM_3D_SUBSPACES;

    int prim_idx = optixGetPrimitiveIndex();
    int entry_id = (prim_idx / 4) % NUM_FINE_PER_SUBSPACE;

    float3 q = params.query_points[logical_ray_id];
    float3 c = params.centroid_xyz[subspace * NUM_FINE_PER_SUBSPACE + entry_id];
    float value = q.x * c.x + q.y * c.y + q.z * c.z;

    // Discard non-positive geometric hits immediately (culls obtuse cone clippings)
    if (value <= 0.0f) {
        optixIgnoreIntersection();
        return;
    }

    unsigned int cnt = optixGetPayload_1();
    optixSetPayload_1(cnt + 1);

    if (cnt < params.max_hits) {
        int base = logical_ray_id * params.max_hits + cnt;
        params.out_hit_centroid[base] = entry_id;
        params.out_hit_value[base]    = value;
    }

    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__corr_3d() {}
