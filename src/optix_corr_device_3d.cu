#include <optix.h>
#include <math.h>
#include "optix_corr_shared_3d.h"

extern "C" __constant__ CorrParams3D params;

// One ray per (query token, subspace). Launch index is subspace-major so neighbouring
// threads trace into the same GAS; outputs use the token-major logical ray id.
extern "C" __global__ void __raygen__corr_3d() {
    uint3 idx = optixGetLaunchIndex();
    int total_rays = optixGetLaunchDimensions().x;
    int nq = total_rays / NUM_3D_SUBSPACES;

    int subspace = idx.x / nq;
    int query_id = idx.x % nq;
    int logical_ray_id = query_id * NUM_3D_SUBSPACES + subspace;

    float3 q = params.query_points[logical_ray_id];
    float qn = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);
    float inv_norm = 1.0f / (qn + 1e-12f);

    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 dir    = make_float3(q.x * inv_norm, q.y * inv_norm, q.z * inv_norm);

    unsigned int p0 = (unsigned int)logical_ray_id;
    unsigned int p1 = 0;
    unsigned int p2 = __float_as_uint(qn);
    OptixTraversableHandle gas = params.subspace_gas[subspace];

    if (params.geometry == GEOM_POLAR) {
        // Codeword c is the plane x.c = 1. A ray from the origin along unit d meets it at
        // t = 1 / (d.c) (only when d.c > 0), so t <= tmax = 1/tau  <=>  d.c >= tau, exactly.
        optixTrace(gas, origin, dir, 0.0f, params.tmax[subspace], 0.0f, 255,
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1, p2);
    } else {
        optixTrace(gas, origin, dir, 0.0f, 2.0f, 0.0f, 255,
                   OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1, p2);
    }
    params.out_hit_count[logical_ray_id] = (int)p1;
}

extern "C" __global__ void __anyhit__corr_3d() {
    int logical_ray_id = (int)optixGetPayload_0();
    int subspace = logical_ray_id % NUM_3D_SUBSPACES;
    int prim_idx = optixGetPrimitiveIndex();
    int E = params.num_entries;

    int entry_id;
    float value;
    if (params.geometry == GEOM_POLAR) {
        // One triangle per codeword; the GAS holds the codewords with |c| > tau sorted by norm.
        entry_id = params.prim_eid[subspace * E + prim_idx];
        // q.c = |q| (d.c) = |q| / t. No codeword load needed.
        value = __uint_as_float(optixGetPayload_2()) / optixGetRayTmax();
    } else {
        entry_id = (prim_idx / 4) % E;
        float3 q = params.query_points[logical_ray_id];
        float3 c = params.centroid_xyz[subspace * E + entry_id];
        value = q.x * c.x + q.y * c.y + q.z * c.z;
        // Discard non-positive geometric hits immediately (culls obtuse cone clippings)
        if (value <= 0.0f) {
            optixIgnoreIntersection();
            return;
        }
    }

    unsigned int cnt = optixGetPayload_1();
    optixSetPayload_1(cnt + 1);

    int base = logical_ray_id * params.max_hits;
    if (cnt < (unsigned int)params.max_hits) {
        params.out_hit_centroid[base + cnt] = entry_id;
        params.out_hit_value[base + cnt]    = value;
    } else {
        // Buffer full: keep the best max_hits hits, not the first ones (any-hit order is
        // arbitrary). Replace the current minimum if this hit beats it.
        int   min_j = 0;
        float min_v = params.out_hit_value[base];
        for (int j = 1; j < params.max_hits; ++j) {
            float v = params.out_hit_value[base + j];
            if (v < min_v) { min_v = v; min_j = j; }
        }
        if (value > min_v) {
            params.out_hit_centroid[base + min_j] = entry_id;
            params.out_hit_value[base + min_j]    = value;
        }
    }

    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__corr_3d() {}
