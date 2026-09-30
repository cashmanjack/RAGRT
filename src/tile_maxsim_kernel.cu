#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32

// Native 192D Ragged MaxSim: 192 dimensions = 96 half2 = 24 float4 loads per token
__global__ void tile_maxsim_192d_kernel(
    const half*    __restrict__ Q,            // [32, 192]
    const half*    __restrict__ all_doc_embs, // [total_tokens, 192] packed ragged
    const int64_t* __restrict__ doc_offsets,  // [num_cands]
    const int*     __restrict__ doc_lens,     // [num_cands]
    int num_cands,
    float*         __restrict__ out_scores    // [num_cands]
) {
    int warp_id  = threadIdx.x / WARP_SIZE;
    int lane_id  = threadIdx.x % WARP_SIZE;
    int cand_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (cand_idx >= num_cands) return;

    // Load query token lane: 96 half2 = 192 dimensions
    half2 q_reg[96];
    const half2* q_ptr = reinterpret_cast<const half2*>(Q + lane_id * 192);
    #pragma unroll
    for (int d = 0; d < 96; ++d) {
        q_reg[d] = q_ptr[d];
    }

    int64_t start = doc_offsets[cand_idx];
    int length    = doc_lens[cand_idx];

    float running_max = -1e9f;

    // Stream 192D tokens using 24 float4 loads per token
    for (int i = 0; i < length; ++i) {
        const float4* d_ptr = reinterpret_cast<const float4*>(
            all_doc_embs + (start + i) * 192);
        
        float dot = 0.0f;
        #pragma unroll
        for (int chunk = 0; chunk < 24; ++chunk) {
            float4 f4 = d_ptr[chunk];
            half2* h2 = reinterpret_cast<half2*>(&f4);

            half2 p0 = __hmul2(q_reg[chunk * 4 + 0], h2[0]);
            half2 p1 = __hmul2(q_reg[chunk * 4 + 1], h2[1]);
            half2 p2 = __hmul2(q_reg[chunk * 4 + 2], h2[2]);
            half2 p3 = __hmul2(q_reg[chunk * 4 + 3], h2[3]);

            dot += __half2float(p0.x) + __half2float(p0.y) +
                   __half2float(p1.x) + __half2float(p1.y) +
                   __half2float(p2.x) + __half2float(p2.y) +
                   __half2float(p3.x) + __half2float(p3.y);
        }
        if (dot > running_max) running_max = dot;
    }

    // Warp-level sum reduction across 32 query tokens
    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        running_max += __shfl_down_sync(0xFFFFFFFF, running_max, off);
    }

    if (lane_id == 0) {
        out_scores[cand_idx] = running_max;
    }
}

// 128D Fallback Kernel
__global__ void tile_maxsim_128d_kernel(
    const half*    __restrict__ Q,
    const half*    __restrict__ all_doc_embs,
    const int64_t* __restrict__ doc_offsets,
    const int*     __restrict__ doc_lens,
    int num_cands,
    float*         __restrict__ out_scores
) {
    int warp_id  = threadIdx.x / WARP_SIZE;
    int lane_id  = threadIdx.x % WARP_SIZE;
    int cand_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (cand_idx >= num_cands) return;

    half2 q_reg[64];
    const half2* q_ptr = reinterpret_cast<const half2*>(Q + lane_id * 128);
    #pragma unroll
    for (int d = 0; d < 64; ++d) q_reg[d] = q_ptr[d];

    int64_t start = doc_offsets[cand_idx];
    int length    = doc_lens[cand_idx];
    float running_max = -1e9f;

    for (int i = 0; i < length; ++i) {
        const float4* d_ptr = reinterpret_cast<const float4*>(all_doc_embs + (start + i) * 128);
        float dot = 0.0f;
        #pragma unroll
        for (int chunk = 0; chunk < 16; ++chunk) {
            float4 f4 = d_ptr[chunk];
            half2* h2 = reinterpret_cast<half2*>(&f4);
            half2 p0 = __hmul2(q_reg[chunk * 4 + 0], h2[0]);
            half2 p1 = __hmul2(q_reg[chunk * 4 + 1], h2[1]);
            half2 p2 = __hmul2(q_reg[chunk * 4 + 2], h2[2]);
            half2 p3 = __hmul2(q_reg[chunk * 4 + 3], h2[3]);
            dot += __half2float(p0.x) + __half2float(p0.y) +
                   __half2float(p1.x) + __half2float(p1.y) +
                   __half2float(p2.x) + __half2float(p2.y) +
                   __half2float(p3.x) + __half2float(p3.y);
        }
        if (dot > running_max) running_max = dot;
    }

    #pragma unroll
    for (int off = 16; off > 0; off /= 2)
        running_max += __shfl_down_sync(0xFFFFFFFF, running_max, off);

    if (lane_id == 0) out_scores[cand_idx] = running_max;
}

extern "C" void launch_tile_maxsim(
    const void* d_Q, const void* d_all_doc_embs, const int64_t* d_doc_offsets,
    const int* d_doc_lens, int num_cands, float* d_out_scores, cudaStream_t stream
) {
    if (num_cands <= 0) return;
    int blocks = (num_cands + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    tile_maxsim_128d_kernel<<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, stream>>>(
        (const half*)d_Q, (const half*)d_all_doc_embs, d_doc_offsets,
        d_doc_lens, num_cands, d_out_scores
    );
}

extern "C" void launch_tile_maxsim_192d(
    const void* d_Q, const void* d_all_doc_embs, const int64_t* d_doc_offsets,
    const int* d_doc_lens, int num_cands, float* d_out_scores, cudaStream_t stream
) {
    if (num_cands <= 0) return;
    int blocks = (num_cands + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    tile_maxsim_192d_kernel<<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, stream>>>(
        (const half*)d_Q, (const half*)d_all_doc_embs, d_doc_offsets,
        d_doc_lens, num_cands, d_out_scores
    );
}
