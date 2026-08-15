#include <optix.h>
#include <optix_function_table.h>
#include <optix_stubs.h>
// FIX: required exactly once, in exactly one .cpp file, to define the
// symbols optix_stubs.h declares. Missing this causes an "undefined
// reference" LINKER error, not a compile error -- easy to lose time on
// if you don't know to look for it.
#include <optix_function_table_definition.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "optix_rag_shared.h"

// ---------------------------------------------------------------------
// Hardcoded test data -- must match validate_optix_rag.py exactly
// ---------------------------------------------------------------------
static SphereSpec h_spheres[NUM_SPHERES] = {
    // subspace 0
    { 0.0f,  0.0f, 0},
    { 0.5f,  0.5f, 0},
    {-0.5f,  0.5f, 0},
    { 0.5f, -0.5f, 0},
    { 1.5f,  0.0f, 0},
    // subspace 1
    { 0.0f,  0.0f, 1},
    {-0.5f, -0.5f, 1},
    { 0.5f, -0.5f, 1},
    {-0.5f,  0.5f, 1},
    { 1.5f,  0.0f, 1},
};

static float2 h_query_subspaces[NUM_SUBSPACES] = {
    {0.2f, 0.1f},
    {-0.2f, -0.2f},
};

// ---------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define OPTIX_CHECK(call)                                                       \
    do {                                                                        \
        OptixResult res__ = (call);                                             \
        if (res__ != OPTIX_SUCCESS) {                                           \
            fprintf(stderr, "OptiX error %s (code %d) at %s:%d\n",             \
                    optixGetErrorString(res__), (int)res__, __FILE__, __LINE__); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static void context_log_cb(unsigned int, const char*, const char* message, void*) {
    fprintf(stderr, "[OptiX] %s\n", message);
}

static bool loadPTX(const char* filename, std::vector<char>& buffer) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return false;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buffer.resize(size + 1);
    if (fread(buffer.data(), 1, size, fp) != (size_t)size) { fclose(fp); return false; }
    buffer[size] = '\0';
    fclose(fp);
    return true;
}

template<typename T>
struct SbtRecord {
    alignas(OPTIX_SBT_RECORD_HEADER_SIZE) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenData {};
struct MissData {};
typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData>   MissSbtRecord;
typedef SbtRecord<HitSbtData> HitSbtRecord;

static OptixTraversableHandle createGAS(OptixDeviceContext context,
                                        std::vector<OptixAabb>& aabbs) {
    OptixAabb* d_aabb = nullptr;
    CUDA_CHECK(cudaMalloc(&d_aabb, aabbs.size() * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(d_aabb, aabbs.data(), aabbs.size() * sizeof(OptixAabb),
                          cudaMemcpyHostToDevice));

    std::vector<uint32_t> sbt_offsets(NUM_SPHERES);
    for (int i = 0; i < NUM_SPHERES; ++i) sbt_offsets[i] = i;
    uint32_t* d_sbt_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sbt_offsets, sbt_offsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_sbt_offsets, sbt_offsets.data(),
                          sbt_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    OptixTraversableHandle gas_handle;

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.numAabbs = (unsigned int)aabbs.size();
    build_input.customPrimitiveArray.aabbBuffers = &d_aabb;
    build_input.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    build_input.customPrimitiveArray.flags = &flags;
    build_input.customPrimitiveArray.numSbtRecords = NUM_SPHERES;
    build_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_offsets;
    build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    build_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &build_input, 1,
                                             &buffer_sizes));

    void* d_temp_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_buffer, buffer_sizes.tempSizeInBytes));
    void* d_output_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output_buffer, buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(context, nullptr, &accel_options, &build_input, 1,
                                d_temp_buffer, buffer_sizes.tempSizeInBytes,
                                d_output_buffer, buffer_sizes.outputSizeInBytes,
                                &gas_handle, nullptr, 0));

    CUDA_CHECK(cudaFree(d_aabb));
    CUDA_CHECK(cudaFree(d_sbt_offsets));
    CUDA_CHECK(cudaFree(d_temp_buffer));
    CUDA_CHECK(cudaFree(d_output_buffer));

    return gas_handle;
}

static void buildHitSbtData(std::vector<HitSbtData>& hit_data) {
    for (int i = 0; i < NUM_SPHERES; ++i) {
        const SphereSpec& s = h_spheres[i];
        HitSbtData d;
        d.center      = make_float3(s.x, s.y, 2.0f * s.subspace_id + 1.0f);
        d.radius      = RADIUS;
        d.subspace_id = s.subspace_id;
        hit_data.push_back(d);
    }
}

int main() {
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context = nullptr;
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions context_options = {};
    context_options.logCallbackFunction = context_log_cb;
    context_options.logCallbackLevel = 3;

    CUcontext cu_ctx = nullptr;
    cuCtxGetCurrent(&cu_ctx);
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &context_options, &context));

    std::vector<char> ptx_buffer;
    if (!loadPTX("optix_rag.ptx", ptx_buffer)) {
        fprintf(stderr, "Failed to load optix_rag.ptx\n");
        return 1;
    }

    OptixModule module = nullptr;
    char log[2048];
    size_t log_size = sizeof(log);
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    // FIX: was 2 + MAX_HITS (7). Now just 2 -- a packed pointer to the
    // RayPayload struct, not the struct's contents spread across registers.
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 0;   // no attributes reported/read
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    OPTIX_CHECK(optixModuleCreate(context, &module_compile_options,
                                  &pipeline_compile_options, ptx_buffer.data(),
                                  ptx_buffer.size(), log, &log_size, &module));

    OptixProgramGroupOptions program_group_options = {};
    char log2[2048];
    size_t log_size2 = sizeof(log2);

    OptixProgramGroup raygen_group = nullptr;
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = module;
    raygen_desc.raygen.entryFunctionName = "__raygen__query";
    OPTIX_CHECK(optixProgramGroupCreate(context, &raygen_desc, 1, &program_group_options,
                                        log2, &log_size2, &raygen_group));

    OptixProgramGroup miss_group = nullptr;
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module;
    miss_desc.miss.entryFunctionName = "__miss__ms";
    log_size2 = sizeof(log2);
    OPTIX_CHECK(optixProgramGroupCreate(context, &miss_desc, 1, &program_group_options,
                                        log2, &log_size2, &miss_group));

    OptixProgramGroup hit_group = nullptr;
    OptixProgramGroupDesc hit_desc = {};
    hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_desc.hitgroup.moduleIS = module;
    hit_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    hit_desc.hitgroup.moduleAH = module;
    hit_desc.hitgroup.entryFunctionNameAH = "__anyhit__record_hits";
    log_size2 = sizeof(log2);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hit_desc, 1, &program_group_options,
                                        log2, &log_size2, &hit_group));

    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;

    OptixProgramGroup program_groups[] = { raygen_group, miss_group, hit_group };
    OPTIX_CHECK(optixPipelineCreate(context, &pipeline_compile_options,
                                    &pipeline_link_options, program_groups, 3,
                                    log, &log_size, &pipeline));

    std::vector<OptixAabb> h_aabbs;
    std::vector<HitSbtData> h_hit_data;
    buildHitSbtData(h_hit_data);

    for (int i = 0; i < NUM_SPHERES; ++i) {
        const HitSbtData& s = h_hit_data[i];
        OptixAabb aabb;
        aabb.minX = s.center.x - s.radius; aabb.minY = s.center.y - s.radius; aabb.minZ = s.center.z - s.radius;
        aabb.maxX = s.center.x + s.radius; aabb.maxY = s.center.y + s.radius; aabb.maxZ = s.center.z + s.radius;
        h_aabbs.push_back(aabb);
    }

    OptixTraversableHandle gas_handle = createGAS(context, h_aabbs);

    RayGenSbtRecord raygen_record;
    memset(&raygen_record, 0, sizeof(raygen_record));
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_group, &raygen_record));
    RayGenSbtRecord* d_raygen_record = nullptr;
    CUDA_CHECK(cudaMalloc(&d_raygen_record, sizeof(RayGenSbtRecord)));
    CUDA_CHECK(cudaMemcpy(d_raygen_record, &raygen_record, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

    MissSbtRecord miss_record;
    memset(&miss_record, 0, sizeof(miss_record));
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_group, &miss_record));
    MissSbtRecord* d_miss_record = nullptr;
    CUDA_CHECK(cudaMalloc(&d_miss_record, sizeof(MissSbtRecord)));
    CUDA_CHECK(cudaMemcpy(d_miss_record, &miss_record, sizeof(MissSbtRecord), cudaMemcpyHostToDevice));

    std::vector<HitSbtRecord> hit_records(NUM_SPHERES);
    for (int i = 0; i < NUM_SPHERES; ++i) {
        memset(&hit_records[i], 0, sizeof(HitSbtRecord));
        OPTIX_CHECK(optixSbtRecordPackHeader(hit_group, &hit_records[i]));
        hit_records[i].data = h_hit_data[i];
    }
    HitSbtRecord* d_hit_records = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hit_records, hit_records.size() * sizeof(HitSbtRecord)));
    CUDA_CHECK(cudaMemcpy(d_hit_records, hit_records.data(),
                          hit_records.size() * sizeof(HitSbtRecord), cudaMemcpyHostToDevice));

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = (CUdeviceptr)d_raygen_record;
    sbt.missRecordBase = (CUdeviceptr)d_miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = (CUdeviceptr)d_hit_records;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitSbtRecord);
    sbt.hitgroupRecordCount = NUM_SPHERES;

    float2* d_query = nullptr;
    CUDA_CHECK(cudaMalloc(&d_query, NUM_SUBSPACES * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(d_query, h_query_subspaces, NUM_SUBSPACES * sizeof(float2), cudaMemcpyHostToDevice));

    float* d_distances = nullptr;
    CUDA_CHECK(cudaMalloc(&d_distances, NUM_SUBSPACES * MAX_HITS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_distances, 0, NUM_SUBSPACES * MAX_HITS * sizeof(float)));

    int* d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, NUM_SUBSPACES * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counts, 0, NUM_SUBSPACES * sizeof(int)));

    LaunchParams params = {};
    params.query_subspaces = d_query;
    params.distances = d_distances;
    params.counts = d_counts;
    params.gas_handle = gas_handle;

    CUdeviceptr d_params = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(LaunchParams)));
    CUDA_CHECK(cudaMemcpy((void*)d_params, &params, sizeof(LaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline, nullptr, d_params, sizeof(LaunchParams), &sbt,
                            NUM_SUBSPACES, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> distances(NUM_SUBSPACES * MAX_HITS);
    std::vector<int> counts(NUM_SUBSPACES);
    CUDA_CHECK(cudaMemcpy(distances.data(), d_distances, NUM_SUBSPACES * MAX_HITS * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts.data(), d_counts, NUM_SUBSPACES * sizeof(int), cudaMemcpyDeviceToHost));

    for (int s = 0; s < NUM_SUBSPACES; ++s) {
        printf("subspace %d: hit_count=%d\n", s, counts[s]);
        for (int i = 0; i < counts[s] && i < MAX_HITS; ++i) {
            printf("  hit %d distance = %.6f\n", i, distances[s * MAX_HITS + i]);
        }
    }

    CUDA_CHECK(cudaFree((void*)d_params));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_raygen_record));
    CUDA_CHECK(cudaFree(d_miss_record));
    CUDA_CHECK(cudaFree(d_hit_records));
    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(raygen_group);
    optixProgramGroupDestroy(miss_group);
    optixProgramGroupDestroy(hit_group);
    optixModuleDestroy(module);
    optixDeviceContextDestroy(context);

    return 0;
}
