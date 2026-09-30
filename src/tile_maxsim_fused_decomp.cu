#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32

__global__ void tile_maxsim_fused_decomp_kernel(
    const half*    __restrict__ Q,                          // [32, 128]
    const int*     __restrict__ pids,                       // [num_cands]
    const int64_t* __restrict__ doc_offsets,                // [num_docs + 1]
    const int*     __restrict__ doc_lens,                   // [num_docs]
    int num_cands,
    const int*     __restrict__ codes,                      // [total_tokens]
    const uint8_t* __restrict__ residuals,                  // [total_tokens, 32]
    const half*    __restrict__ centroids,                  // [num_centroids, 128]
    const half*    __restrict__ bucket_weights,             // [4]
    const uint8_t* __restrict__ reversed_bit_map,           // [256]
    const uint8_t* __restrict__ decomp_table,               // [256, 4]
    float*         __restrict__ out_scores                  // [num_cands]
) {
    __shared__ half s_doc_emb[WARPS_PER_BLOCK][2][128];

    int warp_id  = threadIdx.x / WARP_SIZE;
    int lane_id  = threadIdx.x % WARP_SIZE;
    int cand_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (cand_idx >= num_cands) return;

    half2 q_reg[64];
    const half2* q_ptr = reinterpret_cast<const half2*>(Q + lane_id * 128);
    #pragma unroll
    for (int d = 0; d < 64; ++d) q_reg[d] = q_ptr[d];

    int pid       = pids[cand_idx];
    int64_t start = doc_offsets[pid];
    int length    = doc_lens[pid];

    float running_max = -1e9f;

    auto unpack_token = [&](int64_t t_idx, int buf_idx) {
        int cid = codes[t_idx];
        uint8_t byte_val = residuals[t_idx * 32 + lane_id];
        uint8_t rev_byte = reversed_bit_map[byte_val];

        half d0 = __hadd(centroids[cid * 128 + lane_id * 4 + 0], bucket_weights[decomp_table[rev_byte * 4 + 0]]);
        half d1 = __hadd(centroids[cid * 128 + lane_id * 4 + 1], bucket_weights[decomp_table[rev_byte * 4 + 1]]);
        half d2 = __hadd(centroids[cid * 128 + lane_id * 4 + 2], bucket_weights[decomp_table[rev_byte * 4 + 2]]);
        half d3 = __hadd(centroids[cid * 128 + lane_id * 4 + 3], bucket_weights[decomp_table[rev_byte * 4 + 3]]);

        float fd0 = __half2float(d0), fd1 = __half2float(d1);
        float fd2 = __half2float(d2), fd3 = __half2float(d3);
        float local_sq = fd0 * fd0 + fd1 * fd1 + fd2 * fd2 + fd3 * fd3;

        #pragma unroll
        for (int off = 16; off > 0; off /= 2) local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, off);
        float total_sq = __shfl_sync(0xFFFFFFFF, local_sq, 0);
        float inv_norm = rsqrtf(total_sq + 1e-12f);

        s_doc_emb[warp_id][buf_idx][lane_id * 4 + 0] = __float2half(fd0 * inv_norm);
        s_doc_emb[warp_id][buf_idx][lane_id * 4 + 1] = __float2half(fd1 * inv_norm);
        s_doc_emb[warp_id][buf_idx][lane_id * 4 + 2] = __float2half(fd2 * inv_norm);
        s_doc_emb[warp_id][buf_idx][lane_id * 4 + 3] = __float2half(fd3 * inv_norm);
    };

    if (length > 0) {
        unpack_token(start, 0);
        __syncwarp();

        int cur_buf = 0;
        for (int t = 0; t < length; ++t) {
            int nxt_buf = cur_buf ^ 1;
            if (t + 1 < length) unpack_token(start + t + 1, nxt_buf);

            const float4* doc_ptr = reinterpret_cast<const float4*>(&s_doc_emb[warp_id][cur_buf][0]);
            float dot = 0.0f;
            #pragma unroll
            for (int chunk = 0; chunk < 16; ++chunk) {
                float4 f4 = doc_ptr[chunk];
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
            __syncwarp();
            cur_buf = nxt_buf;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off /= 2) running_max += __shfl_down_sync(0xFFFFFFFF, running_max, off);
    if (lane_id == 0) out_scores[cand_idx] = running_max;
}

extern "C" void launch_tile_maxsim_fused_decomp(
    const void* d_Q, const int* d_pids, const int64_t* d_doc_offsets, const int* d_doc_lens,
    int num_cands, const int* d_codes, const uint8_t* d_residuals, const void* d_centroids,
    const void* d_bucket_weights, const uint8_t* d_reversed_bit_map, const uint8_t* d_decomp_table,
    float* d_out_scores, cudaStream_t stream
) {
    if (num_cands <= 0) return;
    int blocks = (num_cands + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    tile_maxsim_fused_decomp_kernel<<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, stream>>>(
        (const half*)d_Q, d_pids, d_doc_offsets, d_doc_lens, num_cands,
        d_codes, d_residuals, (const half*)d_centroids, (const half*)d_bucket_weights,
        d_reversed_bit_map, d_decomp_table, d_out_scores
    );
}
