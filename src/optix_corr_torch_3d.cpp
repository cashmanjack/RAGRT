#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <optix.h>
#include <optix_function_table.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>
#include "optix_corr_shared_3d.h"

#define CUDA_CHECK(call) do { cudaError_t e=(call); TORCH_CHECK(e==cudaSuccess, "CUDA: ", cudaGetErrorString(e)); } while(0)
#define OPTIX_CHECK(call) do { OptixResult r=(call); TORCH_CHECK(r==OPTIX_SUCCESS, "OptiX: ", optixGetErrorString(r)); } while(0)

static void ctx_log(unsigned int lvl, const char* tag, const char* msg, void*) {}

static bool loadPTX(const std::string& fn, std::vector<char>& buf) {
    FILE* f = fopen(fn.c_str(), "rb"); if (!f) return false;
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    buf.resize(sz+1);
    if (fread(buf.data(),1,sz,f) != (size_t)sz){ fclose(f); return false; }
    buf[sz]='\0'; fclose(f); return true;
}

struct RG {}; struct MS {}; struct HG {};
template<typename T> struct SbtRecord { alignas(OPTIX_SBT_RECORD_HEADER_SIZE) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; T data; };
typedef SbtRecord<RG> RayGenSbt;
typedef SbtRecord<MS> MissSbt;
typedef SbtRecord<HG> HitSbt;

extern "C" void launch_ragrt_fused_stage23_csr256(
    const int* d_out_c, const float* d_out_v, const int* d_out_hit_count,
    const int* d_topc, const float* d_scores,
    int total_rays, int max_hits, int k_eids, int k_cent, int ntok, int num_centroids,
    const int* d_csr_row_ptrs, const uint8_t* d_csr_col_eids,
    const int64_t* d_csr_offsets, const uint16_t* d_csr_lengths, const uint8_t* d_map_packed_24,
    float* d_passage_scores, int num_passages,
    const uint32_t* d_doc_predicates, uint32_t query_mask,
    int buf_idx, cudaStream_t stream
);

extern "C" void launch_rt_maxsim_rescore_from_approx(
    const void* d_approx, const int* d_pids,
    int num_cands, int ntok, int num_passages,
    float* d_out_scores, cudaStream_t stream
);

extern "C" half* ragrt_approx_score_buffer(int buf_idx);
extern "C" int   ragrt_dropped_task_count(int buf_idx);

extern "C" void launch_tile_maxsim_fused_decomp(
    const void* d_Q, const int* d_pids, const int64_t* d_doc_offsets, const int* d_doc_lens,
    int num_cands, const int* d_codes, const uint8_t* d_residuals, const void* d_centroids,
    const void* d_bucket_weights, const uint8_t* d_reversed_bit_map, const uint8_t* d_decomp_table,
    float* d_out_scores, cudaStream_t stream
);

extern "C" void launch_ragrt_stage2_only(
    const int* d_out_c, const float* d_out_v, const int* d_out_hit_count,
    const int* d_topc, const float* d_scores,
    int total_rays, int max_hits, int k_eids, int k_cent, int ntok, int num_centroids,
    const int* d_csr_row_ptrs, const uint8_t* d_csr_col_eids,
    const int64_t* d_csr_offsets, const uint16_t* d_csr_lengths,
    int buf_idx, cudaStream_t stream
);

extern "C" void launch_ragrt_stage3_only(
    const uint8_t* d_map_packed_24, float* d_passage_scores, int num_passages,
    const uint32_t* d_doc_predicates, uint32_t query_mask, int ntok,
    int buf_idx, cudaStream_t stream
);

class CorrIndex3D {
public:
    OptixDeviceContext ctx = nullptr;
    OptixModule mod = nullptr;
    OptixPipeline pipe = nullptr;
    OptixProgramGroup rg_grp=nullptr, ms_grp=nullptr, hg_grp=nullptr;
    OptixShaderBindingTable sbt = {};
    void *d_rg=nullptr,*d_ms=nullptr,*d_hg=nullptr;

    std::vector<OptixTraversableHandle> gas_handles;
    std::vector<void*> gas_out_bufs, gas_v_bufs, gas_i_bufs;
    float3* d_centroid_xyz = nullptr;

    torch::Tensor buf_out_c[2], buf_out_v[2], buf_out_n[2];
    torch::Tensor d_passage_scores[2], d_final_scores[2], d_candidate_pids[2];
    CorrParams3D* d_p_single = nullptr;
    int64_t buf_total_rays = 0;
    int max_allocated_cands = 0;

    cudaStream_t stream_cand = nullptr;
    cudaStream_t stream_score = nullptr;
    cudaEvent_t event_cand_ready[2];
    cudaEvent_t event_tms_done[2];

    bool is_bound = false;
    torch::Tensor b_csr_row_ptrs, b_csr_col_eids, b_csr_offsets, b_csr_lengths, b_map_packed;
    torch::Tensor b_doc_offsets, b_doc_lens, b_codes, b_residuals;
    torch::Tensor b_centroids_128, b_bucket_weights, b_reversed_bit_map, b_decomp_table;
    torch::Tensor b_doc_predicates;
    int64_t b_num_passages = 0;
    int b_num_centroids = 0;

    CorrIndex3D(std::string ptx_path) {
        CUDA_CHECK(cudaFree(0)); OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions copts={}; copts.logCallbackFunction=ctx_log; copts.logCallbackLevel=4;
        CUcontext cu=nullptr; cuCtxGetCurrent(&cu); OPTIX_CHECK(optixDeviceContextCreate(cu,&copts,&ctx));

        std::vector<char> ptx; TORCH_CHECK(loadPTX(ptx_path, ptx), "failed to load candidate PTX");
        OptixModuleCompileOptions mco={}; mco.optLevel=OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        OptixPipelineCompileOptions pco={}; pco.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pco.numPayloadValues=2; // Payload 0: ray_id, Payload 1: private hit counter (zero atomics!)
        pco.numAttributeValues=2;
        pco.pipelineLaunchParamsVariableName="params";
        OPTIX_CHECK(optixModuleCreate(ctx,&mco,&pco,ptx.data(),ptx.size(),nullptr,nullptr,&mod));

        OptixProgramGroupOptions gopts={}; OptixProgramGroupDesc d={};
        d.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d.raygen.module=mod; d.raygen.entryFunctionName="__raygen__corr_3d";
        OPTIX_CHECK(optixProgramGroupCreate(ctx,&d,1,&gopts,nullptr,nullptr,&rg_grp));
        d={}; d.kind=OPTIX_PROGRAM_GROUP_KIND_MISS; d.miss.module=mod; d.miss.entryFunctionName="__miss__corr_3d";
        OPTIX_CHECK(optixProgramGroupCreate(ctx,&d,1,&gopts,nullptr,nullptr,&ms_grp));
        d={}; d.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d.hitgroup.moduleAH=mod; d.hitgroup.entryFunctionNameAH="__anyhit__corr_3d";
        OPTIX_CHECK(optixProgramGroupCreate(ctx,&d,1,&gopts,nullptr,nullptr,&hg_grp));

        OptixPipelineLinkOptions plo={}; plo.maxTraceDepth=1; OptixProgramGroup groups[]={rg_grp,ms_grp,hg_grp};
        OPTIX_CHECK(optixPipelineCreate(ctx,&pco,&plo,groups,3,nullptr,nullptr,&pipe));
        OptixStackSizes ss={}; for(auto g:groups) OPTIX_CHECK(optixUtilAccumulateStackSizes(g,&ss,pipe));
        uint32_t dc_t=0,dc_s=0,cont=0; OPTIX_CHECK(optixUtilComputeStackSizes(&ss,1,0,0,&dc_t,&dc_s,&cont));
        OPTIX_CHECK(optixPipelineSetStackSize(pipe,dc_t,dc_s,cont,2));

        RayGenSbt rg={}; MissSbt ms={}; HitSbt hg={};
        OPTIX_CHECK(optixSbtRecordPackHeader(rg_grp,&rg)); OPTIX_CHECK(optixSbtRecordPackHeader(ms_grp,&ms)); OPTIX_CHECK(optixSbtRecordPackHeader(hg_grp,&hg));
        CUDA_CHECK(cudaMalloc(&d_rg,sizeof(rg))); CUDA_CHECK(cudaMemcpy(d_rg,&rg,sizeof(rg),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_ms,sizeof(ms))); CUDA_CHECK(cudaMemcpy(d_ms,&ms,sizeof(ms),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_hg,sizeof(hg))); CUDA_CHECK(cudaMemcpy(d_hg,&hg,sizeof(hg),cudaMemcpyHostToDevice));
        sbt.raygenRecord=(CUdeviceptr)d_rg; sbt.missRecordBase=(CUdeviceptr)d_ms; sbt.missRecordStrideInBytes=sizeof(ms); sbt.missRecordCount=1;
        sbt.hitgroupRecordBase=(CUdeviceptr)d_hg; sbt.hitgroupRecordStrideInBytes=sizeof(hg); sbt.hitgroupRecordCount=1;

        CUDA_CHECK(cudaMalloc(&d_p_single, sizeof(CorrParams3D)));

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_cand, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_score, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event_cand_ready[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&event_tms_done[b], cudaEventDisableTiming));
        }
    }

    ~CorrIndex3D() {
        if (d_p_single) cudaFree(d_p_single);
        if (d_centroid_xyz) cudaFree(d_centroid_xyz);
        if (stream_cand) cudaStreamDestroy(stream_cand);
        if (stream_score) cudaStreamDestroy(stream_score);
        for (int b = 0; b < 2; ++b) {
            if (event_cand_ready[b]) cudaEventDestroy(event_cand_ready[b]);
            if (event_tms_done[b]) cudaEventDestroy(event_tms_done[b]);
        }
    }

    void ensure_cand_capacity(int required_cands) {
        if (required_cands > max_allocated_cands) {
            int new_capacity = std::max(16384, required_cands * 2);
            for (int b = 0; b < 2; ++b) {
                d_candidate_pids[b] = torch::zeros({new_capacity}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
                d_final_scores[b]   = torch::empty({new_capacity}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            }
            max_allocated_cands = new_capacity;
        }
    }

    void free_scene() {
        for(auto p:gas_out_bufs) if(p) cudaFree(p);
        for(auto p:gas_v_bufs) if(p) cudaFree(p);
        for(auto p:gas_i_bufs) if(p) cudaFree(p);
        gas_out_bufs.clear(); gas_v_bufs.clear(); gas_i_bufs.clear(); gas_handles.clear();
        if(d_centroid_xyz) cudaFree(d_centroid_xyz);
        d_centroid_xyz=nullptr;
    }

    void build(torch::Tensor codebooks, double base_radius = 0.95, int64_t num_tris = 4) {
        TORCH_CHECK(num_tris == 4, "build() requires num_tris == 4; got ", num_tris);
        free_scene();
        int S = NUM_3D_SUBSPACES;
        int E = NUM_FINE_PER_SUBSPACE;

        auto cba = codebooks.accessor<float, 3>();
        std::vector<float3> h_centroid_xyz(S * E);
        gas_handles.resize(S);
        constexpr float PI = 3.14159265358979323846f;

        for (int s = 0; s < S; ++s) {
            std::vector<float3> verts;
            std::vector<uint3> indices;

            for (int e = 0; e < E; ++e) {
                float cx = cba[s][e][0], cy = cba[s][e][1], cz = cba[s][e][2];
                h_centroid_xyz[s * E + e] = make_float3(cx, cy, cz);

                float norm = sqrtf(cx*cx + cy*cy + cz*cz);
                float radius = sqrtf(base_radius * base_radius + norm * norm);

                float3 n = (norm > 1e-5f) ? make_float3(cx/norm, cy/norm, cz/norm) : make_float3(0, 0, 1);
                float3 u = (fabs(n.x) > 0.1f) ? make_float3(-n.y, n.x, 0.0f) : make_float3(0.0f, -n.z, n.y);
                float un = sqrtf(u.x*u.x + u.y*u.y + u.z*u.z);
                u.x /= un; u.y /= un; u.z /= un;
                float3 v = make_float3(n.y*u.z - n.z*u.y, n.z*u.x - n.x*u.z, n.x*u.y - n.y*u.x);

                uint32_t b = (uint32_t)verts.size();
                int P = (int)num_tris;
                verts.push_back(make_float3(cx, cy, cz));
                for (int k = 0; k < P; ++k) {
                    float theta = k * (2.0f * PI / (float)P);
                    verts.push_back(make_float3(cx + radius * cosf(theta)*u.x + radius * sinf(theta)*v.x,
                                                cy + radius * cosf(theta)*u.y + radius * sinf(theta)*v.y,
                                                cz + radius * cosf(theta)*u.z + radius * sinf(theta)*v.z));
                }
                for (int k = 0; k < P; ++k) indices.push_back(make_uint3(b, b + 1 + k, b + 1 + ((k + 1) % P)));
            }

            float3* d_v = nullptr; uint3* d_i = nullptr; void* out = nullptr; void* tmp = nullptr;
            CUDA_CHECK(cudaMalloc(&d_v, verts.size() * sizeof(float3)));
            CUDA_CHECK(cudaMalloc(&d_i, indices.size() * sizeof(uint3)));
            CUDA_CHECK(cudaMemcpy(d_v, verts.data(), verts.size() * sizeof(float3), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_i, indices.data(), indices.size() * sizeof(uint3), cudaMemcpyHostToDevice));

            OptixBuildInput bi = {}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            CUdeviceptr dp_v = (CUdeviceptr)d_v;
            bi.triangleArray.vertexBuffers = &dp_v;
            bi.triangleArray.numVertices = (unsigned)verts.size();
            bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            bi.triangleArray.indexBuffer = (CUdeviceptr)d_i;
            bi.triangleArray.numIndexTriplets = (unsigned)indices.size();
            bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            static const unsigned int fl = OPTIX_GEOMETRY_FLAG_NONE;
            bi.triangleArray.flags = &fl; bi.triangleArray.numSbtRecords = 1;

            OptixAccelBuildOptions ao = {}; ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
            OptixAccelBufferSizes bs; OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
            CUDA_CHECK(cudaMalloc(&tmp, bs.tempSizeInBytes)); CUDA_CHECK(cudaMalloc(&out, bs.outputSizeInBytes));
            OPTIX_CHECK(optixAccelBuild(ctx, nullptr, &ao, &bi, 1, (CUdeviceptr)tmp, bs.tempSizeInBytes, (CUdeviceptr)out, bs.outputSizeInBytes, &gas_handles[s], nullptr, 0));
            CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaFree(tmp));
            gas_out_bufs.push_back(out); gas_v_bufs.push_back(d_v); gas_i_bufs.push_back(d_i);
        }

        CUDA_CHECK(cudaMalloc(&d_centroid_xyz, h_centroid_xyz.size() * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(d_centroid_xyz, h_centroid_xyz.data(), h_centroid_xyz.size() * sizeof(float3), cudaMemcpyHostToDevice));
    }

    void bind_index(
        torch::Tensor csr_row_ptrs, torch::Tensor csr_col_eids,
        torch::Tensor csr_offsets, torch::Tensor csr_lengths, torch::Tensor map_packed,
        torch::Tensor doc_offsets, torch::Tensor doc_lens,
        torch::Tensor codes, torch::Tensor residuals, torch::Tensor centroids_128,
        torch::Tensor bucket_weights, torch::Tensor reversed_bit_map, torch::Tensor decomp_table,
        torch::Tensor doc_predicates, int64_t num_passages, int num_centroids
    ) {
        b_csr_row_ptrs = csr_row_ptrs.to(torch::kInt32).contiguous();
        b_csr_col_eids = csr_col_eids.contiguous();
        b_csr_offsets = csr_offsets.contiguous();
        b_csr_lengths = csr_lengths.contiguous();
        b_map_packed = map_packed.contiguous();
        b_doc_offsets = doc_offsets.contiguous();
        b_doc_lens = doc_lens.contiguous();
        b_codes = codes.contiguous();
        b_residuals = residuals.contiguous();
        b_centroids_128 = centroids_128.to(torch::kFloat16).contiguous();
        b_bucket_weights = bucket_weights.contiguous();
        b_reversed_bit_map = reversed_bit_map.contiguous();
        b_decomp_table = decomp_table.contiguous();
        b_doc_predicates = doc_predicates.to(torch::kUInt32).contiguous();
        b_num_passages = doc_predicates.size(0);
        b_num_centroids = num_centroids;

        for (int b = 0; b < 2; ++b) {
            d_passage_scores[b] = torch::zeros({b_num_passages}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        }
        ensure_cand_capacity(16384);
        is_bound = true;
    }

    torch::Tensor search_single_query_native(
        torch::Tensor Q_full_128_fp32, torch::Tensor Q_full_fp16, torch::Tensor Q_sub3d, torch::Tensor topc, torch::Tensor scores,
        int k_cent, int k_candidates, int k_top, int maxsim_backend = 0, uint32_t query_mask = 0, int k_eids = 16
    ) {
        TORCH_CHECK(is_bound, "Index must be bound before search!");
        int nq = (int)Q_sub3d.size(0);
        ensure_cand_capacity(k_candidates);
        int total_rays = nq * NUM_3D_SUBSPACES;
        auto qp = Q_sub3d.view({total_rays, 3}).contiguous();

        if (total_rays > buf_total_rays) {
            auto oi = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            auto of = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            for (int b = 0; b < 2; ++b) {
                buf_out_c[b] = torch::zeros({total_rays, 128}, oi);
                buf_out_v[b] = torch::zeros({total_rays, 128}, of);
                buf_out_n[b] = torch::zeros({total_rays}, oi);
            }
            buf_total_rays = total_rays;
        } else {
            buf_out_c[0].zero_(); buf_out_v[0].zero_(); buf_out_n[0].zero_();
        }

        CorrParams3D p = {};
        p.query_points = (float3*)qp.data_ptr<float>();
        for (int s = 0; s < NUM_3D_SUBSPACES; ++s) p.subspace_gas[s] = gas_handles[s];
        p.centroid_xyz = d_centroid_xyz;
        p.out_hit_centroid = buf_out_c[0].data_ptr<int>();
        p.out_hit_value = buf_out_v[0].data_ptr<float>();
        p.out_hit_count = buf_out_n[0].data_ptr<int>();
        p.max_hits = 128;

        CUDA_CHECK(cudaMemcpy(d_p_single, &p, sizeof(CorrParams3D), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(pipe, nullptr, (CUdeviceptr)d_p_single, sizeof(CorrParams3D), &sbt, total_rays, 1, 1));

        d_passage_scores[0].zero_();

        launch_ragrt_fused_stage23_csr256(
            buf_out_c[0].data_ptr<int>(), buf_out_v[0].data_ptr<float>(), buf_out_n[0].data_ptr<int>(),
            topc.data_ptr<int>(), scores.data_ptr<float>(),
            total_rays, 128, k_eids, k_cent, nq, b_num_centroids,
            b_csr_row_ptrs.data_ptr<int>(), (const uint8_t*)b_csr_col_eids.data_ptr(),
            b_csr_offsets.data_ptr<int64_t>(), (const uint16_t*)b_csr_lengths.data_ptr(), b_map_packed.data_ptr<uint8_t>(),
            d_passage_scores[0].data_ptr<float>(), (int)b_num_passages,
            (const uint32_t*)b_doc_predicates.data_ptr(), query_mask, 0, 0
        );

        auto topk_cands = torch::topk(d_passage_scores[0], k_candidates);
        auto candidate_pids = std::get<1>(topk_cands).to(torch::kInt32).contiguous();

        if (maxsim_backend == 0) {
            launch_tile_maxsim_fused_decomp(
                Q_full_fp16.data_ptr(), candidate_pids.data_ptr<int>(), b_doc_offsets.data_ptr<int64_t>(), b_doc_lens.data_ptr<int>(),
                k_candidates, b_codes.data_ptr<int>(), b_residuals.data_ptr<uint8_t>(), b_centroids_128.data_ptr(),
                b_bucket_weights.data_ptr(), b_reversed_bit_map.data_ptr<uint8_t>(), b_decomp_table.data_ptr<uint8_t>(),
                d_final_scores[0].data_ptr<float>(), 0
            );
        } else {
            launch_rt_maxsim_rescore_from_approx(
                ragrt_approx_score_buffer(0), candidate_pids.data_ptr<int>(),
                k_candidates, nq, (int)b_num_passages,
                d_final_scores[0].data_ptr<float>(), 0
            );
        }

        auto topk_final = torch::topk(d_final_scores[0].slice(0, 0, k_candidates), std::min(k_top, k_candidates));
        return candidate_pids.index({std::get<1>(topk_final)});
    }

    std::tuple<torch::Tensor, std::vector<float>> search_single_query_profiled(
        torch::Tensor Q_full_128_fp32, torch::Tensor Q_full_fp16, torch::Tensor Q_sub3d, torch::Tensor topc, torch::Tensor scores,
        int k_cent, int k_candidates, int k_top, int maxsim_backend = 0, uint32_t query_mask = 0, int k_eids = 16
    ) {
        TORCH_CHECK(is_bound, "Index must be bound before search!");
        int nq = (int)Q_sub3d.size(0);
        ensure_cand_capacity(k_candidates);
        int total_rays = nq * NUM_3D_SUBSPACES;
        auto qp = Q_sub3d.view({total_rays, 3}).contiguous();

        if (total_rays > buf_total_rays) {
            auto oi = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            auto of = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            for (int b = 0; b < 2; ++b) {
                buf_out_c[b] = torch::zeros({total_rays, 128}, oi);
                buf_out_v[b] = torch::zeros({total_rays, 128}, of);
                buf_out_n[b] = torch::zeros({total_rays}, oi);
            }
            buf_total_rays = total_rays;
        } else {
            buf_out_c[0].zero_(); buf_out_v[0].zero_(); buf_out_n[0].zero_();
        }

        CorrParams3D p = {};
        p.query_points = (float3*)qp.data_ptr<float>();
        for (int s = 0; s < NUM_3D_SUBSPACES; ++s) p.subspace_gas[s] = gas_handles[s];
        p.centroid_xyz = d_centroid_xyz;
        p.out_hit_centroid = buf_out_c[0].data_ptr<int>();
        p.out_hit_value = buf_out_v[0].data_ptr<float>();
        p.out_hit_count = buf_out_n[0].data_ptr<int>();
        p.max_hits = 128;

        cudaEvent_t ev0, ev1, ev2, ev3, ev4;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2);
        cudaEventCreate(&ev3); cudaEventCreate(&ev4);

        // Stage 1: OptiX Ray Tracing
        cudaEventRecord(ev0, 0);
        CUDA_CHECK(cudaMemcpy(d_p_single, &p, sizeof(CorrParams3D), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(pipe, nullptr, (CUdeviceptr)d_p_single, sizeof(CorrParams3D), &sbt, total_rays, 1, 1));
        cudaEventRecord(ev1, 0);

        // Stage 2: CSR Gather Tasks
        launch_ragrt_stage2_only(
            buf_out_c[0].data_ptr<int>(), buf_out_v[0].data_ptr<float>(), buf_out_n[0].data_ptr<int>(),
            topc.data_ptr<int>(), scores.data_ptr<float>(),
            total_rays, 128, k_eids, k_cent, nq, b_num_centroids,
            b_csr_row_ptrs.data_ptr<int>(), (const uint8_t*)b_csr_col_eids.data_ptr(),
            b_csr_offsets.data_ptr<int64_t>(), (const uint16_t*)b_csr_lengths.data_ptr(),
            0, 0
        );
        cudaEventRecord(ev2, 0);

        // Stage 3: Cooperative Passage Scoring & Sum Reduction
        d_passage_scores[0].zero_();
        launch_ragrt_stage3_only(
            b_map_packed.data_ptr<uint8_t>(), d_passage_scores[0].data_ptr<float>(), (int)b_num_passages,
            (const uint32_t*)b_doc_predicates.data_ptr(), query_mask, nq, 0, 0
        );
        auto topk_cands = torch::topk(d_passage_scores[0], k_candidates);
        auto candidate_pids = std::get<1>(topk_cands).to(torch::kInt32).contiguous();
        cudaEventRecord(ev3, 0);

        // Stage 4: TileMaxSim Rerank
        launch_tile_maxsim_fused_decomp(
            Q_full_fp16.data_ptr(), candidate_pids.data_ptr<int>(), b_doc_offsets.data_ptr<int64_t>(), b_doc_lens.data_ptr<int>(),
            k_candidates, b_codes.data_ptr<int>(), b_residuals.data_ptr<uint8_t>(), b_centroids_128.data_ptr(),
            b_bucket_weights.data_ptr(), b_reversed_bit_map.data_ptr<uint8_t>(), b_decomp_table.data_ptr<uint8_t>(),
            d_final_scores[0].data_ptr<float>(), 0
        );
        auto topk_final = torch::topk(d_final_scores[0].slice(0, 0, k_candidates), std::min(k_top, k_candidates));
        cudaEventRecord(ev4, 0);

        cudaEventSynchronize(ev4);
        float t_s1 = 0, t_s2 = 0, t_s3 = 0, t_s4 = 0;
        cudaEventElapsedTime(&t_s1, ev0, ev1);
        cudaEventElapsedTime(&t_s2, ev1, ev2);
        cudaEventElapsedTime(&t_s3, ev2, ev3);
        cudaEventElapsedTime(&t_s4, ev3, ev4);

        cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2);
        cudaEventDestroy(ev3); cudaEventDestroy(ev4);

        return std::make_tuple(candidate_pids.index({std::get<1>(topk_final)}), std::vector<float>{t_s1, t_s2, t_s3, t_s4});
    }

    torch::Tensor search_batch_pipelined(
        std::vector<torch::Tensor> Q_full_128_list,
        std::vector<torch::Tensor> Q_full_fp16_list,
        std::vector<torch::Tensor> Q_sub3d_list,
        std::vector<torch::Tensor> topc_list,
        std::vector<torch::Tensor> scores_list,
        int k_cent, int k_candidates, int k_top,
        int maxsim_backend = 0, uint32_t query_mask = 0, int k_eids = 16
    ) {
        TORCH_CHECK(is_bound, "Index must be bound before search!");
        ensure_cand_capacity(k_candidates);
        int num_queries = (int)Q_full_fp16_list.size();
        auto out_results = torch::empty({num_queries, k_top}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        int max_total_rays = 32 * NUM_3D_SUBSPACES;
        if (max_total_rays > buf_total_rays) {
            auto oi = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            auto of = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            for (int b = 0; b < 2; ++b) {
                buf_out_c[b] = torch::zeros({max_total_rays, 128}, oi);
                buf_out_v[b] = torch::zeros({max_total_rays, 128}, of);
                buf_out_n[b] = torch::zeros({max_total_rays}, oi);
            }
            buf_total_rays = max_total_rays;
        }

        std::vector<CorrParams3D> h_params_all(num_queries);
        for (int i = 0; i < num_queries; ++i) {
            int buf_idx = i % 2;
            h_params_all[i].query_points = (float3*)Q_sub3d_list[i].data_ptr<float>();
            for (int s = 0; s < NUM_3D_SUBSPACES; ++s) h_params_all[i].subspace_gas[s] = gas_handles[s];
            h_params_all[i].centroid_xyz = d_centroid_xyz;
            h_params_all[i].out_hit_centroid = buf_out_c[buf_idx].data_ptr<int>();
            h_params_all[i].out_hit_value = buf_out_v[buf_idx].data_ptr<float>();
            h_params_all[i].out_hit_count = buf_out_n[buf_idx].data_ptr<int>();
            h_params_all[i].max_hits = 128;
        }
        auto d_params_batch = torch::empty({num_queries * (int64_t)sizeof(CorrParams3D)}, 
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        CUDA_CHECK(cudaMemcpy(d_params_batch.data_ptr(), h_params_all.data(), 
            num_queries * sizeof(CorrParams3D), cudaMemcpyHostToDevice));

        c10::cuda::CUDAStream c_stream_cand  = c10::cuda::getStreamFromExternal(stream_cand, 0);
        c10::cuda::CUDAStream c_stream_score = c10::cuda::getStreamFromExternal(stream_score, 0);

        for (int i = 0; i < num_queries; ++i) {
            int buf_idx = i % 2;
            if (i >= 2) CUDA_CHECK(cudaStreamWaitEvent(stream_cand, event_tms_done[buf_idx], 0));

            int nq = (int)Q_sub3d_list[i].size(0);
            int total_rays = nq * NUM_3D_SUBSPACES;
            CUdeviceptr param_ptr = (CUdeviceptr)(d_params_batch.data_ptr<uint8_t>() + i * sizeof(CorrParams3D));

            {
                c10::cuda::CUDAStreamGuard guard(c_stream_cand);
                buf_out_c[buf_idx].zero_(); buf_out_v[buf_idx].zero_(); buf_out_n[buf_idx].zero_();
                OPTIX_CHECK(optixLaunch(pipe, stream_cand, param_ptr, sizeof(CorrParams3D), &sbt, total_rays, 1, 1));
                d_passage_scores[buf_idx].zero_();

                launch_ragrt_fused_stage23_csr256(
                    buf_out_c[buf_idx].data_ptr<int>(), buf_out_v[buf_idx].data_ptr<float>(), buf_out_n[buf_idx].data_ptr<int>(),
                    topc_list[i].data_ptr<int>(), scores_list[i].data_ptr<float>(),
                    total_rays, 128, k_eids, k_cent, nq, b_num_centroids,
                    b_csr_row_ptrs.data_ptr<int>(), (const uint8_t*)b_csr_col_eids.data_ptr(),
                    b_csr_offsets.data_ptr<int64_t>(), (const uint16_t*)b_csr_lengths.data_ptr(), b_map_packed.data_ptr<uint8_t>(),
                    d_passage_scores[buf_idx].data_ptr<float>(), (int)b_num_passages,
                    (const uint32_t*)b_doc_predicates.data_ptr(), query_mask, buf_idx, stream_cand
                );

                auto topk_cands = torch::topk(d_passage_scores[buf_idx], k_candidates);
                d_candidate_pids[buf_idx].slice(0, 0, k_candidates).copy_(std::get<1>(topk_cands));
                CUDA_CHECK(cudaEventRecord(event_cand_ready[buf_idx], stream_cand));
            }

            {
                c10::cuda::CUDAStreamGuard guard(c_stream_score);
                CUDA_CHECK(cudaStreamWaitEvent(stream_score, event_cand_ready[buf_idx], 0));

                launch_tile_maxsim_fused_decomp(
                    Q_full_fp16_list[i].data_ptr(), d_candidate_pids[buf_idx].data_ptr<int>(),
                    b_doc_offsets.data_ptr<int64_t>(), b_doc_lens.data_ptr<int>(),
                    k_candidates, b_codes.data_ptr<int>(), b_residuals.data_ptr<uint8_t>(), b_centroids_128.data_ptr(),
                    b_bucket_weights.data_ptr(), b_reversed_bit_map.data_ptr<uint8_t>(), b_decomp_table.data_ptr<uint8_t>(),
                    d_final_scores[buf_idx].data_ptr<float>(), stream_score
                );

                auto topk_final = torch::topk(d_final_scores[buf_idx].slice(0, 0, k_candidates), std::min(k_top, k_candidates));
                auto final_ranked = d_candidate_pids[buf_idx].slice(0, 0, k_candidates).index({std::get<1>(topk_final)});
                out_results[i].copy_(final_ranked);
                CUDA_CHECK(cudaEventRecord(event_tms_done[buf_idx], stream_score));
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_cand));
        CUDA_CHECK(cudaStreamSynchronize(stream_score));
        return out_results;
    }
};

torch::Tensor native_tile_maxsim_fused_decomp(
    torch::Tensor Q, torch::Tensor pids, torch::Tensor doc_offsets, torch::Tensor doc_lens,
    torch::Tensor codes, torch::Tensor residuals, torch::Tensor centroids,
    torch::Tensor bucket_weights, torch::Tensor reversed_bit_map, torch::Tensor decomp_table
) {
    TORCH_CHECK(Q.dim() == 2 && Q.size(0) == 32 && Q.size(1) == 128,
        "fused MaxSim kernel reads Q as [32, 128]; got [", Q.size(0), ", ", Q.size(1), "]");
    int num_cands = pids.size(0);
    auto out_scores = torch::empty({num_cands}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    launch_tile_maxsim_fused_decomp(
        Q.data_ptr(), pids.data_ptr<int>(), doc_offsets.data_ptr<int64_t>(), doc_lens.data_ptr<int>(),
        num_cands, codes.data_ptr<int>(), residuals.data_ptr<uint8_t>(), centroids.data_ptr(),
        bucket_weights.data_ptr(), reversed_bit_map.data_ptr<uint8_t>(), decomp_table.data_ptr<uint8_t>(),
        out_scores.data_ptr<float>(), 0
    );
    return out_scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<CorrIndex3D>(m, "CorrIndex3D")
        .def(pybind11::init<std::string>(), pybind11::arg("ptx_path"))
        .def("build", &CorrIndex3D::build, pybind11::arg("codebooks"), pybind11::arg("base_radius") = 0.95, pybind11::arg("num_tris") = 4)
        .def("bind_index", &CorrIndex3D::bind_index)
        .def("search_single_query_profiled", &CorrIndex3D::search_single_query_profiled,
             pybind11::arg("Q_full_128_fp32"), pybind11::arg("Q_full_fp16"), pybind11::arg("Q_sub3d"), pybind11::arg("topc"), pybind11::arg("scores"),
             pybind11::arg("k_cent"), pybind11::arg("k_candidates"), pybind11::arg("k_top"), pybind11::arg("maxsim_backend") = 0, pybind11::arg("query_mask") = 0, pybind11::arg("k_eids") = 16)     
        .def("search_single_query_native", &CorrIndex3D::search_single_query_native,
             pybind11::arg("Q_full_128_fp32"), pybind11::arg("Q_full_fp16"), pybind11::arg("Q_sub3d"), pybind11::arg("topc"), pybind11::arg("scores"),
             pybind11::arg("k_cent"), pybind11::arg("k_candidates"), pybind11::arg("k_top"), pybind11::arg("maxsim_backend") = 0, pybind11::arg("query_mask") = 0, pybind11::arg("k_eids") = 16)
        .def("search_batch_pipelined", &CorrIndex3D::search_batch_pipelined,
             pybind11::arg("Q_full_128_list"), pybind11::arg("Q_full_fp16_list"), pybind11::arg("Q_sub3d_list"), pybind11::arg("topc_list"), pybind11::arg("scores_list"),
             pybind11::arg("k_cent"), pybind11::arg("k_candidates"), pybind11::arg("k_top"), pybind11::arg("maxsim_backend") = 0, pybind11::arg("query_mask") = 0, pybind11::arg("k_eids") = 16);
    m.def("native_tile_maxsim_fused_decomp", &native_tile_maxsim_fused_decomp);
}
