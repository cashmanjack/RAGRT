#include <optix.h>
#include "optix_rag_shared.h"

// Named launch-params constant -- matches
// pipeline_compile_options.pipelineLaunchParamsVariableName = "params"
// set on the host side. OptiX populates this automatically each launch;
// no manual optixGetLaunchParams() cast needed.
__constant__ LaunchParams params;

// ---------------------------------------------------------------------
// Pointer <-> payload-register packing.
// FIX: the previous version tried to pass a RayPayload struct directly
// as an optixTrace() argument and read it back via a nonexistent
// optixGetPayloadPointer() -- neither is real OptiX 7.x API. Payloads
// are individual 32-bit registers; the standard pattern for carrying an
// arbitrary/growable struct is to pass its pointer split across two
// registers, exactly as NVIDIA's own OptiX SDK samples do.
// ---------------------------------------------------------------------
static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = static_cast<uint32_t>(uptr >> 32);
    i1 = static_cast<uint32_t>(uptr & 0xFFFFFFFFu);
}

static __forceinline__ __device__ void* unpackPointer(uint32_t i0, uint32_t i1) {
    const uint64_t uptr = (static_cast<uint64_t>(i0) << 32) | static_cast<uint64_t>(i1);
    return reinterpret_cast<void*>(uptr);
}

static __forceinline__ __device__ RayPayload* getPRD() {
    return reinterpret_cast<RayPayload*>(
        unpackPointer(optixGetPayload_0(), optixGetPayload_1()));
}

extern "C" __global__ void __raygen__query() {
    uint3 idx = optixGetLaunchIndex();
    int s = idx.x;
    if (s >= NUM_SUBSPACES) return;

    RayPayload payload;
    payload.subspace_id = s;
    payload.hit_count   = 0;
    for (int i = 0; i < MAX_HITS; ++i) payload.hit_ts[i] = 0.0f;

    uint32_t p0, p1;
    packPointer(&payload, p0, p1);

    float2 q      = params.query_subspaces[s];
    float3 origin = make_float3(q.x, q.y, 2.0f * s);
    float3 dir    = make_float3(0.0f, 0.0f, 1.0f);

    // tmax = 1.0: matches JUNO's convention exactly. The ray physically
    // cannot reach the next subspace's z-layer -- this is a geometric
    // guarantee, not just the any-hit shader's subspace_id filter below.
    optixTrace(
        params.gas_handle,
        origin, dir,
        0.0f, 1.0f, 0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0, 0, 0,
        p0, p1);

    for (int i = 0; i < payload.hit_count; ++i) {
        float t = payload.hit_ts[i];
        // JUNO L2 reconstruction: d = sqrt(R^2 - (1 - t_hit)^2).
        // No sphere-coordinate memory access.
        float d = sqrtf(fmaxf(0.0f, RADIUS * RADIUS - (1.0f - t) * (1.0f - t)));
        params.distances[s * MAX_HITS + i] = d;
    }
    params.counts[s] = payload.hit_count;
}





extern "C" __global__ void __intersection__sphere() {
    HitSbtData* sbt = reinterpret_cast<HitSbtData*>(optixGetSbtDataPointer());

    float3 origin = optixGetObjectRayOrigin();
    float3 dir    = optixGetObjectRayDirection();
    float3 center = sbt->center;
    float  R      = sbt->radius;

    // Explicit scalar math to avoid relying on external operator overload headers
    float3 oc = make_float3(origin.x - center.x, origin.y - center.y, origin.z - center.z);
    
    float a = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
    float b = 2.0f * (oc.x * dir.x + oc.y * dir.y + oc.z * dir.z);
    float c = (oc.x * oc.x + oc.y * oc.y + oc.z * oc.z) - R * R;

    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return;

    float sqrt_disc = sqrtf(disc);
    float t0 = (-b - sqrt_disc) / (2.0f * a);
    float t1 = (-b + sqrt_disc) / (2.0f * a);

    float t = (t0 >= 0.0f) ? t0 : ((t1 >= 0.0f) ? t1 : -1.0f);
    if (t < 0.0f || t > optixGetRayTmax()) return;

    optixReportIntersection(t, 0);
}





extern "C" __global__ void __anyhit__record_hits() {
    RayPayload* payload = getPRD();
    HitSbtData* sbt = reinterpret_cast<HitSbtData*>(optixGetSbtDataPointer());

    // Only accept spheres in the ray's own JUNO subspace.
    if (sbt->subspace_id != payload->subspace_id) {
        optixIgnoreIntersection();
        return;
    }

    if (payload->hit_count < MAX_HITS) {
        payload->hit_ts[payload->hit_count] = optixGetRayTmax();
        payload->hit_count++;
    }
}

extern "C" __global__ void __miss__ms() {}
