// Fused decompress + MaxSim rerank (Stage 4; also used by PLAID+TMS).
//
//   score(doc) = sum over the 32 query rows r of max over doc tokens t of q_r . d_t / |d_t|
//
// Two kernels, selected at runtime (ragrt_set_rerank_mode):
//   0 "simt": warp per candidate, lane = query row, half2 products on CUDA cores (original).
//   1 "wmma": warp per candidate, 16 doc tokens per tile, Q.D^T on tensor cores
//             (m16n16k16, fp16 in, fp32 accumulate). Doc tokens are decompressed without
//             normalizing; the 1/|d_t| scale is applied to the fp32 dot products.
// Candidates with pid < 0 (padding) score -inf.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <math.h>

#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32

static int g_rerank_mode = 1;
extern "C" void ragrt_set_rerank_mode(int mode) { g_rerank_mode = mode; }
extern "C" int  ragrt_get_rerank_mode() { return g_rerank_mode; }

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
    __shared__ half s_doc_emb[WARPS_PER_BLOCK][2][136];

    int warp_id  = threadIdx.x / WARP_SIZE;
    int lane_id  = threadIdx.x % WARP_SIZE;
    int cand_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (cand_idx >= num_cands) return;
    int pid = pids[cand_idx];
    if (pid < 0) { if (lane_id == 0) out_scores[cand_idx] = -INFINITY; return; }

    half2 q_reg[64];
    const half2* q_ptr = reinterpret_cast<const half2*>(Q + lane_id * 128);
    #pragma unroll
    for (int d = 0; d < 64; ++d) q_reg[d] = q_ptr[d];

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
                dot += __half2float(__low2half(p0)) + __half2float(__high2half(p0)) +
                       __half2float(__low2half(p1)) + __half2float(__high2half(p1)) +
                       __half2float(__low2half(p2)) + __half2float(__high2half(p2)) +
                       __half2float(__low2half(p3)) + __half2float(__high2half(p3));
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

// ---------------------------------------------------------------------------
// Tensor-core version
// ---------------------------------------------------------------------------
#define TC_WARPS 8
#define TC_TILE 16
#define TC_LDS 136          // halves per token row in shared memory (128 + 8 pad; multiple of 8)

__global__ void __launch_bounds__(TC_WARPS * 32, 2)
tile_maxsim_wmma_kernel(
    const half*    __restrict__ Q, const int* __restrict__ pids,
    const int64_t* __restrict__ doc_offsets, const int* __restrict__ doc_lens, int num_cands,
    const int*     __restrict__ codes, const uint8_t* __restrict__ residuals,
    const half*    __restrict__ centroids, const half* __restrict__ bucket_weights,
    const uint8_t* __restrict__ reversed_bit_map, const uint8_t* __restrict__ decomp_table,
    float*         __restrict__ out_scores
) {
    using namespace nvcuda;
    __shared__ __align__(32) half s_tile[TC_WARPS][TC_TILE * TC_LDS];   // reused as the fp32 [32][16] result
    __shared__ float s_inv[TC_WARPS][TC_TILE];
    __shared__ uint8_t s_rbm[256];
    __shared__ uint32_t s_dtab[256];
    __shared__ half s_bw[4];
    __shared__ __align__(32) half s_q[32 * TC_LDS];

    for (int i = threadIdx.x; i < 32 * 16; i += blockDim.x) {      // Q: 32 rows x 16 uint4
        int r = i / 16, c = i % 16;
        *reinterpret_cast<uint4*>(s_q + r * TC_LDS + 8 * c) = reinterpret_cast<const uint4*>(Q + r * 128)[c];
    }
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_rbm[i] = reversed_bit_map[i];
        s_dtab[i] = (uint32_t)decomp_table[4 * i] | ((uint32_t)decomp_table[4 * i + 1] << 8) |
                    ((uint32_t)decomp_table[4 * i + 2] << 16) | ((uint32_t)decomp_table[4 * i + 3] << 24);
    }
    if (threadIdx.x < 4) s_bw[threadIdx.x] = bucket_weights[threadIdx.x];
    __syncthreads();

    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int cand = blockIdx.x * TC_WARPS + warp;
    if (cand >= num_cands) return;
    int pid = pids[cand];
    if (pid < 0) { if (lane == 0) out_scores[cand] = -INFINITY; return; }
    int64_t start = doc_offsets[pid];
    int L = doc_lens[pid];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a0, a1;
    half* tile = s_tile[warp];
    float* acc_s = reinterpret_cast<float*>(tile);
    float run_max = -1e9f;                       // lane = query row
    const int t = lane >> 1, hsel = lane & 1;   // decompress: token t, dims [64 hsel, 64 hsel + 64)

    for (int base = 0; base < L; base += TC_TILE) {
        int ti = base + t;
        half* dst = tile + t * TC_LDS + hsel * 64;
        float ss = 0.0f;
        if (ti < L) {
            int64_t tok = start + ti;
            int cid = codes[tok];
            uint4 r4 = *reinterpret_cast<const uint4*>(residuals + tok * 32 + hsel * 16);
            const uint8_t* rb = reinterpret_cast<const uint8_t*>(&r4);
            const uint4* cp = reinterpret_cast<const uint4*>(centroids + (int64_t)cid * 128 + hsel * 64);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {          // 8 dims per step = 2 residual bytes
                uint4 c4 = cp[j];
                const half* ch = reinterpret_cast<const half*>(&c4);
                uint32_t i0 = s_dtab[s_rbm[rb[2 * j]]], i1 = s_dtab[s_rbm[rb[2 * j + 1]]];
                uint4 o4;
                half* oh = reinterpret_cast<half*>(&o4);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    half d0 = __hadd(ch[u],     s_bw[(i0 >> (8 * u)) & 0xFF]);
                    half d1 = __hadd(ch[4 + u], s_bw[(i1 >> (8 * u)) & 0xFF]);
                    float f0 = __half2float(d0), f1 = __half2float(d1);
                    ss += f0 * f0 + f1 * f1;
                    oh[u] = d0; oh[4 + u] = d1;
                }
                *reinterpret_cast<uint4*>(dst + 8 * j) = o4;
            }
        } else {
            uint4 z = make_uint4(0, 0, 0, 0);
            #pragma unroll
            for (int j = 0; j < 8; ++j) *reinterpret_cast<uint4*>(dst + 8 * j) = z;
        }
        ss += __shfl_xor_sync(0xFFFFFFFF, ss, 1);
        if (hsel == 0) s_inv[warp][t] = (ti < L) ? rsqrtf(ss + 1e-12f) : 0.0f;
        __syncwarp();

        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
        wmma::fill_fragment(c0, 0.0f);
        wmma::fill_fragment(c1, 0.0f);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            wmma::load_matrix_sync(b, tile + k * 16, TC_LDS);   // element (dim, token) at token * LDS + dim
            wmma::load_matrix_sync(a0, s_q + k * 16, TC_LDS);
            wmma::load_matrix_sync(a1, s_q + 16 * TC_LDS + k * 16, TC_LDS);
            wmma::mma_sync(c0, a0, b, c0);
            wmma::mma_sync(c1, a1, b, c1);
        }
        __syncwarp();
        wmma::store_matrix_sync(acc_s, c0, 16, wmma::mem_row_major);            // rows 0-15
        wmma::store_matrix_sync(acc_s + 16 * 16, c1, 16, wmma::mem_row_major);  // rows 16-31
        __syncwarp();
        int nvalid = min(TC_TILE, L - base);
        for (int n = 0; n < nvalid; ++n)
            run_max = fmaxf(run_max, acc_s[lane * 16 + n] * s_inv[warp][n]);
        __syncwarp();
    }

    #pragma unroll
    for (int off = 16; off > 0; off /= 2) run_max += __shfl_down_sync(0xFFFFFFFF, run_max, off);
    if (lane == 0) out_scores[cand] = run_max;
}

extern "C" void launch_tile_maxsim_fused_decomp(
    const void* d_Q, const int* d_pids, const int64_t* d_doc_offsets, const int* d_doc_lens,
    int num_cands, const int* d_codes, const uint8_t* d_residuals, const void* d_centroids,
    const void* d_bucket_weights, const uint8_t* d_reversed_bit_map, const uint8_t* d_decomp_table,
    float* d_out_scores, cudaStream_t stream
) {
    if (num_cands <= 0) return;
    if (g_rerank_mode == 1) {
        int blocks = (num_cands + TC_WARPS - 1) / TC_WARPS;
        tile_maxsim_wmma_kernel<<<blocks, TC_WARPS * 32, 0, stream>>>(
            (const half*)d_Q, d_pids, d_doc_offsets, d_doc_lens, num_cands,
            d_codes, d_residuals, (const half*)d_centroids, (const half*)d_bucket_weights,
            d_reversed_bit_map, d_decomp_table, d_out_scores);
    } else {
        int blocks = (num_cands + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        tile_maxsim_fused_decomp_kernel<<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, stream>>>(
            (const half*)d_Q, d_pids, d_doc_offsets, d_doc_lens, num_cands,
            d_codes, d_residuals, (const half*)d_centroids, (const half*)d_bucket_weights,
            d_reversed_bit_map, d_decomp_table, d_out_scores);
    }
}
