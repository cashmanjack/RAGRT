# **JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via Ray Tracing Core** 

ZIHAN LIU, Shanghai Jiao Tong University, Shanghai, China WENTAO NI, Shanghai Jiao Tong University, Shanghai, China JINGWEN LENG, Shanghai Jiao Tong University, Shanghai, China YU FENG, Shanghai Jiao Tong University, Shanghai, China CONG GUO, Shanghai Jiao Tong University, Shanghai, China QUAN CHEN, Shanghai Jiao Tong University, Shanghai, China CHAO LI, Shanghai Jiao Tong University, Shanghai, China MINYI GUO, Shanghai Jiao Tong University, Shanghai, China YUFEI MA, Information Science Academy CETC, Beijing, China FENG ZHANG, Information Science Academy CETC, Beijing, China YUN LIANG, Peking University, Beijing, China 

Approximate Nearest Neighbor Search (ANNS) is a fundamental technique in modern intelligent applications, including recommendation systems and vector databases. With the advent of large language models (LLMs), ANNS plays a critical role in enabling attention pruning mechanism that exploit the sparsity of attention, such as top-K attention and retrieval attention. As a result, the efficiency of ANNS has become increasingly crucial. In this article, we identify a key inefficiency in state-of-the art ANNS methods based on product quantization: the redundant computation and accumulation of pairwise distance with codebook. To address this, we propose JUNO++, the system consists of (i) an end-to-end ANNS search pipeline based on ray-tracing core leveraging sparsity-aware algorithm and ii) an integration of the ray-tracing based ANNS search pipeline to the attention computation. For ANNS search pipeline, evaluation on four datasets indicate 2.2x to 8.5x search throughput improvement. For ANNS-powered sparse attention, JUNO++ achieves a 46% reduction in latency of 𝑞× 𝑘<sup>⊤</sup> calculation comparing to the baseline with almost identical accuracy, which is not only a key component of retrieval-based sparse attention, but also the dominant component in long-context scenario, implying a considerable end-to-end improvement. 

CCS Concepts: • **Computing methodologies** → **Ray tracing** ; • **Information systems** → _Top-k retrieval in databases_ ; **Nearest-neighbor search;** 

This work was supported by the National Natural Science Foundation of China (NSFC) Grants (U21B2017 and 62222210) and Shanghai Qi Zhi Institute Innovation Program SQZ202316. 

Authors’ Contact Information: Zihan Liu, Shanghai Jiao Tong University, Shanghai, China; e-mail: altair.liu@sjtu.edu.cn; Wentao Ni, Shanghai Jiao Tong University, Shanghai, China; e-mail: wennitao@sjtu.edu.cn; Jingwen Leng (corresponding author), Shanghai Jiao Tong University, Shanghai, China; e-mail: leng-jw@cs.sjtu.edu.cn; Yu Feng, Shanghai Jiao Tong University, Shanghai, China; e-mail: y-feng@sjtu.edu.cn; Cong Guo, Shanghai Jiao Tong University, Shanghai, China; e-mail: guocong@sjtu.edu.cn; Quan Chen, Shanghai Jiao Tong University, Shanghai, China; e-mail: chen-quan@cs.sjtu.edu.cn; Chao Li, Shanghai Jiao Tong University, Shanghai, China; e-mail: lichao@cs.sjtu.edu.cn; Minyi Guo (corresponding author), Shanghai Jiao Tong University, Shanghai, China; e-mail: guo-my@cs.sjtu.edu.cn; Yufei Ma, Information Science Academy CETC, Beijing, China; e-mail: mayufei_cxy@163.com; Feng Zhang, Information Science Academy CETC, Beijing, China; e-mail: zhangfeng@cetc.com.cn; Yun Liang, Peking University, Beijing, China; e-mail: ericlyun@pku.edu.cn. 

This work is licensed under a Creative Commons Attribution 4.0 International License. 

© 2025 Copyright held by the owner/author(s). ACM 1544-3973/2025/12-ART133 https://doi.org/10.1145/3768585 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:2 

Additional Key Words and Phrases: Approximate kNN, ray tracing, optix 

### **ACM Reference Format:** 

Zihan Liu, Wentao Ni, Jingwen Leng, Yu Feng, Cong Guo, Quan Chen, Chao Li, Minyi Guo, Yufei Ma, Feng Zhang, and Yun Liang. 2025. JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via Ray Tracing Core. _ACM Trans. Arch. Code Optim._ 22, 4, Article 133 (December 2025), 25 pages. https://doi.org/10.1145/3768585 

## **1 Introduction** 

Embedding vector as a central data representation lies in a crucial position of current intelligent applications [3, 14, 22, 23, 38, 39, 45, 55, 62, 67, 68]. Embedding vectors are typically derived from raw modalities such as text, images, etc. through learning process. Embedding vectors often facilitate nearest neighbor search (NNS) to enable retrieval based on similarity. Typically embedding vectors are high-dimensional, ranging from several dozen to thousands of dimension. This high dimensionality poses significant challenges for exact NNS, making approximate NNS (ANNS) a practical alternative widely adopted in industry [7, 11, 19, 35, 66]. ANNS offer a trade-off between recall and efficiency. 

In the era of transformer-based large language models (LLMs) [52, 61], ANNS also plays a critical role. In addition to retrieval augmented generation (RAG) [59] that utilize ANNS to select proper knowledge domain, ANNS is also employed in generic LLM computation process to enable sparse attention mechanism [8, 32, 36], since the multiplication between query and key projections is actually retrieving the most relevant keys correspond to a query in respect of inner-product. While providing algorithmic prototypes to verify, currently these work fail to bring the speed-up to practice, partly due to inefficient ANNS implementation. 

Product quantization (PQ) is among the most widely adopted approaches for ANNS [31, 35], and often integrated with other optimizations like inverted file index (IVF), graph-based indexing (HNSW, NSG). In this framework, PQ encodes the projected vectors into compact codes within each subspace using pre-trained codebooks offline. During querying, the nearest neighbors are identified by aggregating distance contributions from all subspaces. However, this online computation entails a substantial number of pairwise distance evaluations between query vectors and database entries across low-dimensional subspaces, along with repeated codebook lookups to obtain total distances. 

In this study, we leverage the state-of-the-art FAISS framework [31] to evaluate the efficiency of typical PQ-based ANNS process. Despite utilizing hundreds of codebook entries to quantize vectors in each subspace, only a small subset contributes to the top-100 retrieved results for a given query. In certain subspaces, all top-100 candidates are encoded with a single codebook entry. This codebook entry level sparsity opens the door to optimization by eliminating unnecessary pairwise distance computations and avoiding redundant distance lookups and accumulations for entries not involved in the search results. Furthermore, our analysis reveals that frequently used codebook entries exhibit strong spatial locality, which enhances the benefits of exploiting sparsity. Although these entries may be scattered across memory space, they tend to cluster tightly in the Euclidean space. In certain subspaces, selecting the nearest 25% of entries is sufficient to cover all those used in encoding the top-100 nearest neighbors for a given query. This observation indicates that only entries in close proximity to the query projections are critical for accurate retrieval. 

To harness the observed sparsity and spatial locality, we introduce a selective codebook construction algorithm aimed at accelerating high-dimensional ANNS. Our method adaptively determines a distance threshold within each subspace to retain only the most relevant codebook entries. We observe a strong correlation between this threshold and the local density of search points, which we leverage by training a lightweight offline regression model that takes density as input. At runtime, 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:3 

the model predicts an appropriate threshold, enabling the selection of a small subset of relevant entries for efficient query processing. Our sparsity-aware ANN search technique relies heavily on distance comparison operations, which align well with the capabilities of ray tracing (RT) cores available in modern GPUs [47, 48, 50]. These RT cores efficiently perform intersection tests using tree-based structures [16, 63], achieving logarithmic time complexity. Consequently, RT cores can rapidly identify objects intersected by a given ray [46]. This hardware functionality closely mirrors the core idea of our algorithm: locating nearby codebook entries for each query projection within a subspace. By conceptualizing codebook entries as geometric objects and query projections as rays, we can effectively exploit RT cores to detect proximate entries through intersection. 

We introduce JUNO++, a high-performance ANNS system that integrates algorithmic enhancement with RT core acceleration to exploit sparsity and spatial locality. Despite the advantages of using RT cores, incorporating them into advanced ANN algorithms presents several challenges. First, even after filtering, distance computations for the selected codebook entries remain necessary. Second, directly implementing an adaptive dynamic radius within RT cores can lead to significant runtime overhead due to repeated scene preparation. To address these challenges, we leverage the concept of “time” in ray tracing. Specifically, we use the RT core’s hit time to compute distances efficiently, thereby reducing the need for costly global memory accesses. Furthermore, we map dynamic distance thresholds to ray travel time, eliminating the need for online scene reconstruction.In addition, we enhance JUNO++ to support inner product similarity without the need for additional embedding dimensions, as required by some prior methods. We also implement efficient pipelining between RT and Tensor Cores [51]. 

With proposed algorithmic enhancement and corresponding ray-tracing core acceleration, we provide a solution to accelerated ANNS-powered sparse attention aligned with RetrievalAttention [36]. To facilitate ANNS in attention calculation, we propose i) a PQ-based quantization mechanism for key cache and ii) a accuracy remedy mechanism for higher search speed and quality specially designed for inner-product metric. Then, we illustrate our overall workflow and provide several interfaces for users to construct their own customized workflow for different requirements. This solution significantly reduces the computational load related to the primary bottleneck–KV cache access [56]. 

We assess the performance of JUNO++ across multiple datasets ranging in size from 1 million to 100 million, using both L2 distance and inner product similarity metrics. Our method achieves, on average, a 4.4× improvement (up to 8.5×) for low-precision search and a 2.1× improvement (up to 3.2×) for high-precision search compared to the baseline. Moreover, these gains are constrained by the performance of the RT cores. In large language models (LLMs), JUNO++ reduces the latency of the 𝑄× 𝐾<sup>⊤</sup> operation by nearly 50% in long-context scenarios (e.g., 1M token). 

We make the following main contributions in this work: 

- We study the inefficiency of the typical approximate nearest neighbor search (ANNS) pipeline and identify sparsity and spatial locality in codebook usage. 

- We design a ray tracing hardware based acceleration algorithm for ANNS to leverage the identified sparsity and locality. 

- We integrate proposed ray tracing based ANNS optimizations to large language inference pipeline and accelerate the ANNS-based sparse attention mechanism. 

- We evaluate our method with existing ANNS framework and LLM inference framework, and achieve significant improvement, with detailed breakdown and analysis. 

## **2 Background** 

This section outlines the standard ANN search process, the ray tracing pipeline, and their application to 2D/3D ANN search, along with a discussion on sparse attention. 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:4 





Fig. 1. (left) The example of offline and online parts of PQ in ANNS. (right) The RT pipeline for RT core. 

## **2.1 Approximate Nearest Neighbor Search and Product Quantization** 

The goal of NNS is to find the top-k most similar points to a query in a given dataset. Similarity is commonly measured using L2 distance or the inner product. L2 distance, which follows a loweris-better criterion, is widely used for image similarity measurement [65]. The inner product, on the other hand, operates on a higher-is-better basis and is commonly employed in LLMs and Transformer architectures [52, 57]. 

Exact NNS is computationally expensive and therefore slow. In many practical scenarios, a certain level of search approximation is acceptable, which can be leveraged to enhance throughput. This approach is known as ANNS [54]. One of the most important tecuniques in ANNS is PQ, implemented in top-performing frameworks like FAISS [31] and ScANN [24] together with various indexing techniques. Other indexing and encoding techniques exist, which we will discuss in Section 7. Typically, PQ is combined with IVF, a very straightforward indexing techniques that cluster the search points into many groups and choose several groups as candidates, which is a very simple phase. In this work, we focus on and mainly optimize the PQ process. We use an example to describe PQ process in Figure 1 (left). 

**_Offline Phase_** Given a set of search points, in this example, there are 𝑁 points with 20 dimensions. PQ first splits them along dimensions into several sub-spaces with identical dimensions, in this example, they are split into four sub-spaces (red, green, blue, and yellow) each with 5 dimensions. 1 In every sub-space, a codebook with much less entries (𝑀 in this example) is trained, typically via k-means. 2 Then, for one search point, its projection in a sub-space is encoded to an index, which is the index of the closest entry to the projection in corresponding codebook. In this example, the projection of 14th search point in the green space is closest to the 2<sup>𝑛𝑑</sup> entry in the green codebook. Finally, we get a 𝑁 encoded points with dimension equals to sub-space numbers. 

**_Online Search._** With offline trained codebooks and encode points, we can serve the incoming queries. One a query arrives, it will also be split into identical sub-spaces as offline did. 3 Then, in every sub-space, the query’s projection calculates the distance between all codebook entries and together they form a distance look-up table. 4 Finally, PQ use encoded points as indices to look-up and accumulate the total distance. For example, if a search point is encoded to (3, 2, 2, 1) in four sub-spaces, respectively. The query will looks up for the 3<sup>𝑟𝑑</sup> , 2<sup>𝑛𝑑</sup> , 2<sup>𝑛𝑑</sup> , 1<sup>𝑠𝑡</sup> distance in four distance look-up tables and accumulates to a final approximate distance for this search point (marked in bold). Once the total distances for all encoded search points are computed, the query sorts the results and selects the top-k nearest neighbors. It is important to recognize that ANN searches can produce both false positives and false negatives. 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:5 

## **2.2 Accelerating NN Search with Ray Tracing Core** 

RT is a rendering pipeline distinct from rasterization [25, 42]. In RT, rays are projected from a camera through each pixel into a virtual scene, where they interact with objects to generate pixels with precise colors. However, the RT process can be computationally intensive and time-consuming, as it requires tracking ray-object interactions, leading to large-scale operations. To address this, NVIDIA developed GPUs with specialized hardware RT accelerators, known as RT cores [47]. The RT core employs a BVH tree-based RT algorithm in hardware, which efficiently determines intersections between rays and surfaces in 3D space. Figure 1 (right) demonstrates the typical BVH tree-based RT pipeline. 

NVIDIA’s RT cores consist of two key hardware components designed to accelerate the typical RT pipeline: the **Axis-Aligned Bounding Box** ( **AABB** ) intersection test and the **Bounding Volume Hierarchy** ( **BVH** ) traversal. The AABB-based intersection test begins by creating bounding boxes (aligned with the 𝑥-axes, 𝑦-axes, or 𝑧-axes) to enclose objects that will be tested for ray intersection. The ray then performs a simple interval-based calculation to determine if it intersects the bounding boxes. If an intersection occurs, the ray further checks for intersections with the objects within the intersected box; otherwise, all objects inside that box are disregarded. It is important to note that an AABB can recursively contain smaller AABBs, forming a tree-like structure with logarithmic depth relative to the total number of objects. In this structure, each node represents an AABB, and its child nodes represent smaller contained AABBs. This tree, known as the BVH, enables recursive intersection checks. Given the potentially vast number of conditions and divergences in tree traversal, NVIDIA’s RT cores include specialized hardware to accelerate the BVH traversal process. 

Researchers have leveraged the ability of RT cores to detect intersections in three-dimensional Euclidean space for two-dimensional and three-dimensional NNS [73]. The core idea is to first arrange 𝑁 search points as 𝑁 spheres in the 𝑥𝑂𝑦 plane, each with a radius 𝑟. The queries are then represented as rays originating from the query point and directed along the z-axis, as shown in Figure 1 (right). If a ray intersects any circles, it indicates that the distance between the query (represented by the ray) and the search points (represented by the circles) is less than 𝑟, suggesting that the intersected circles may contain the nearest neighbors of the query. For instance, in Figure 1 (right), ray 𝑞 intersects the two circles in the lower-left quadrant, implying that these circles are potential nearest neighbors of the query. 

The previously mentioned approach using RT cores to accelerate NNS is constrained to lowdimensional spaces (2D and 3D), limiting its practical applicability. In contrast, our work seeks to investigate the efficient use of RT cores for more general NNS tasks, specifically targeting **approximate nearest neighbor** ( **ANN** ) search in high-dimensional spaces. 

## **2.3 Sparse Attention in LLMs** 

LLMs adopting transformer architecture have achieved great success in modern intelligent applications. The core of transformer is multi-head attention that execute multiple parallel attention processes. It can be mathematically described as follows: 

- MHA(𝑄, 𝐾, 𝑉) = Concat(head1, … , headℎ)𝑊𝑂, head𝑖 = Attn(𝑄, 𝐾, 𝑉= 𝐻< 𝑊𝑄, 𝑊𝐾, 𝑊𝑉 >), Attn(𝑄, 𝐾, 𝑉) = softmax(𝑄𝐾<sup>𝑇</sup> /√𝑑𝑘)𝑉. 

Here, 𝑊𝑂,𝑄,𝐾,𝑉 are parameter metrices and 𝐻 is the hidden state. The 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 is applied over the result of inner-product between query and keys. A key and emerging application of ANNS in modern LLMs is sparse attention [8, 32, 36], particularly in long input sequences. In this variant, only the largest inner-product between query and keys are passed to the following 𝑠𝑜𝑓𝑡𝑚𝑎𝑥, while the others are ignored to significantly reduce computational cost. To identify these largest 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:6 



<!-- Start of picture text -->
(a) (b) (c) (d)<br>1E3 Max<br>Indexing Stage Mean<br>PQ: LUT Calculate 90% of top-100 included<br>1E2 PQ: Accumulate<br>1E1<br>1E0 Median<br>Sort by L2(entry, query projection) Q1 / Q3<br>1E- 1 0 64 128 192 256 0 16 32 48 closest medium farthest<br>Recall Codebook entries ID Subspace ID Codebook Entries<br>0.5 100%<br>0.25 50%<br>Subspace ID<br>Entries usage ratio<br>Time for 10k queries (ms) 0.0 % of top-100 included 0%<br>70 % 80 % 90 % 95 % 97 % 99 % 99.9 % 99.99 %<br><!-- End of picture text -->

Fig. 2. (a) Execution time breakdown of searching queries in DEEP1M using FAISS. (b) Codebook entries usage of a single query: higher usage frequency leads to darker color. (c) Max used ratio of codebook entries on every sub-dimension of 100 queries. (d) CDF of entries to contain top-100 from closest to farthest. Ploted with DEEP1M. 

inner-product, ANNS is employed, with most of the existing approaches leveraging FAISS for the search. Specifically, ANNS replaces the traditional 𝑄𝐾<sup>𝑇</sup> process, which involves computing the inner product between query and keys. By focusing only on the largest weights, a softmax function is applied to compute approximate attention scores, which are then used to retrieve necessary values for following calculations. 

## **3 Motivation** 

This section presents an analysis of FAISS [31], a leading library for high-dimensional ANNS. We first examine the execution time breakdown of FAISS queries, then identify and analyze the inefficiencies. Based on these findings, we propose potential optimization strategies. 

## **3.1 Product Quantization Dominance the Execution Time** 

We utilize the latest version of FAISS [31] and the DEEP1M dataset [5] in this study. Specifically, we configure FAISS with IVF, a widely utilized accelerating index. In this configuration, 1,000,000 search points are clustered into 4,096 groups, and the 96-dimensional space is partitioned into 48 2-dimensional sub-spaces and conduct PQ. We measure the execution time of indexing part and PQ related parts using an NVIDIA Geforce RTX 4090 GPU [49]. The experimental results in Figure 2(a) show that the indexing part consume one to three magnitude less time than PQ related part (including distance look up table calculation and total distance accumulation). Furthermore, the time for PQ stages increases with the recall increases. In contrast, the IVF stage remains relatively constant. This is because achieving higher recall necessitates considering more clusters during the IVF stage, which, in turn, involves more codebooks (and their entries) in pairwise distance computations and increases the number of search points contributing to distance accumulation. Collectively, these factors lead to increased time spent in the PQ stages. This observation motivates us to focus on optimizing the PQ related stages, which we will analyze for inefficiencies in the following section. 

## **3.2 Sparsity of Codebook Entries Used by Top-k Neighbors** 

The current ANNS implementation, such as FAISS, computes the pairwise distance between the query projection and codebook entries across all sub-spaces. However, our analysis shows that only a small subset of codebook entries is necessary to identify the top-100 closest search points for a given query. To demonstrate this, we calculate the frequency with which each codebook entry is used by the top-100 search points. The results, presented as a heatmap in Figure 2(b), show statistics for all codebook entries across different sub-spaces. The shading of each cell indicates the frequency of usage, ranging from 0 to 100, with darker colors representing higher usage. For example, a value 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:7 

of 𝑐𝑒𝑙𝑙[42, 81] = 89 means that in the 42<sup>𝑛𝑑</sup> sub-space, 89 of the top-100 search points are encoded using the 81th codebook entry. A zero indicates that none of the top-100 search points are encoded with that codebook entry. 

The utilization of codebook entries in each sub-space for the DEEP1M dataset [5] is shown in Figure 2x, revealing that, on average, only 25% (up to 30%) of the entries are used. To explore how data distribution affects this sparsity, we analyze the SIFT1M [30] and TTI1M[30] datasets. And these datasets also demonstrate an average codebook entry utilization of less than 30%. Leveraging this sparsity could lead to a significant reduction of up to three quarters in floating-point operations required for pairwise distance calculations per query, greatly accelerating the distance look up table calculating. Although, in theory, more computation leads to higher recall, unnecessary calculations can be safely eliminated. From these observations, we derive the first takeaway: **Codebook entries used by top-100 neighbours are sparse.** 

## **3.3 Spatial Locality of Codebook Entries Used by Top-k Neighbors** 

While we have demonstrated the sparsity in the codebook entries used by the top-k points, converting this sparsity into tangible performance improvements may still be challenging, as sparsity often leads to irregular access patterns. However, as shown in Figure 2(b), the codebook entries that are utilized are clustered in the front half of the frequency heatmap. This indicates that the entries used are closer to the query projection, as the heatmap is organized based on the distance between the entry and the query projection in each sub-space. 

To validate this, we compute and plot the **cumulative distribution function** ( **CDF** ) of the top-100 search points, ranging from the closest to the farthest entries in each sub-space. As shown in Figure 2(d), we find that using less than half of the codebook entries allows us to capture over 90% of the top-100 search points. We further examine the SIFT1M and TTI1M datasets and find that although the patterns differ, both datasets exhibit a similar trend, with approximately 50% of the closest entries accounting for over 90% of the top-100 ground truth. Based on this analysis, we derive another key takeaway: **Codebook entries used by top-100 neighbours are closely distributed in the space.** 

## **4 Accelerating ANNS via Sparsity-aware Algorithm and Ray Tracing Core Mapping** 

This section outlines our algorithmic enhancement and corresponding RT core implementation. We first describe details of our adjustments in original algorithms, followed by how we implement the enhanced algorithm on RT core. Finally, we list several RT core specific optimizations. 

## **4.1 Sparsity-based Algorithmic Enhancement** 

In this subsection, we outline the design of our algorithm. The new algorithm is intended to eliminate redundant pair-wise distance calculation when constructing distance look-up table ( 3 in Section 3), offering substantial savings while maintaining an acceptable level of search quality. 

**_Selecting Necessary Entries with Radius_** _._ The intuition of our algorithmic enhancement is to abandon the codebook entries being far away to the query projection in every sub-spaces, and the remained entries are referred to as “selected”. Then, the distance between selected entries and the query projections are computed to construct the distance look-up table for later distance accumulation. To accomplish this, we first need to setup a threshold, or radius, to define being far away and being close to. The detail of how to setup a proper threshold will be discussed in next paragraph. Then, to avoid brute-force pair-wise distance calculation in this process, we group entries into boxes, and bound these boxes with larger boxes and organize them hierarchically into a tree structure. Now we can traverse this tree in a binary-search manner to find close entries to 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:8 



<!-- Start of picture text -->
(a) (b)<br>150 Q0 / Q4Q1 / Q3 100% 99%90%<br>Mean 75%<br>100 50%<br>25%<br>50 0% MeanQ1 / Q3<br>1E0 1E3 1E6 0.0 0.25 0.5 0.75 1.0<br>Point density Radius scaling factor<br>Amount of top-100<br>Radius containing top-100<br><!-- End of picture text -->



Fig. 3. (a) The relation between threshold to contain top-100 search points and region density a query falls into. (b) The amount of top-100 search points contained when threshold scales smaller. _*Q0/Q4_ = _Q1/Q3_ ∓1.5× _IQR, IQR_ = _Q3_ − _Q1._ (c) Overview of JUNO++ search pipeline. 

the query projection in 𝑙𝑜𝑔𝐸 complexity: being far away to a box, being far away to everything inside it. 

**_Determining Proper Radius based on Datasets_** _._ It is vital to setup an appropriate radius in our enhancement, as it determine whether an entry will be finally considered by a query. A stringent radius may exclude too many valid neighbors, while a lenient one could include too many unnecessary entries, wasting time and resources. To identify the proper radius, we perform an in-depth analysis of the relationship between query and the radius that can encompass the top-100 neighbors in every sub-spaces. As shown in Figure 3(a), there is a strong inverse relationship between the radius to include top-100 neighbors and the density where the query projection locates. To compute density, we divide the sub-space into a 100×100 grid, and the density is defined as the ratio of search point projections within the cell to its area. 



Fig. 4. Intuitive illustration of our pipeline. 

Upon this finding, we propose to dynamically determine the proper radius at runtime. We first setup a 100×100 density map and a regression model offline, and train the regression model with several randomly selected search points (density as input and radius to include top-100 neighbors as output). During runtime, we calculate a radius with arrived query. Once the radius is set, we abandon codebook entries beyond this radius as mentioned in the last paragraph. Additionally, we observe a power-law when changing the radius, as shown in Figure 3(b). About 90% of the top-100 search points can be capture with only half of the radius. This suggests that smaller radius can be employed to prune more codebook entries to trade-off 

for higher search performance. To accommodate difference scenarios, we offer users the flexibility to adjust the radius via a dedicated interface to balance between search quality and performance. 

## **4.2 Ray Tracing Core-based Implementation** 

Building on the algorithmic enhancement discussed above, we introduce the details to accelerate the enhanced algorithm via hardware RT. The implementation consists of offline preparation phase and online searching phase. Figure 3(c) shows the overview of our implementation. As shown in Figure 4, 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:9 

intuitively, we place spheres of identical radius–each representing a codebook entry–into the RT scene, and launch rays from the query position to test for intersections with the spheres, thereby determining the proximity of codebook entries to the query. The hardware-provided concepts of ray traversal and termination time can be leveraged for various optimizations. The following contents formally describe these processes. 

**_Offline Preparation_** _._ In the offline phase, we prepare a traversable scene and sub-space level indices from entries to search point projections based on trained codebook of PQ. The details are outlined in Algorithm 1. Notice that we still use PQ with inverted file index as baseline, so we include the IVF part in the presented algorithm. First, we obtain 𝐶 cluster centroids and labels for the IVFPQ to conduct initialize filtering process (Line 2). Then, we generate the codebook via the residuals between search points and their closest centriods among 𝐶 (Line 3–5). Specially, we split the residuals into several 2-dimensional sub-spaces and conduct k-means in these spaces to get codebook entries. In addition, we also maintain indices from one codebook entry to search points use it, for later distance accumulation (Line 8–9). For instance, 𝑀𝑎𝑝[114, 19, 24] includes all search points that meed the following criteria: (i) the search points belongs to the 114th (among 𝐶) IVF cluster, ii) its projections is encoded with the 24th codebook entry in the 19th sub-space. 

Now, we build the traversable scene to utilize RT core with trained codebook (Line 6–7). In the 𝑑<sup>𝑡ℎ</sup> sub-space, we create spheres representing the codebook entries in this sub-space at the (𝑥, 𝑦) coordinates. For the 𝑧-coordinate, we place entries from the same sub-space at 𝑧= 2𝑑+ 1. This arrangement eliminates interference from ray originating in other sub-spaces. The rationale is that the rays, spheres, and their corresponding RT operations for each sub-space are confined to the region {(𝑥, 𝑦, 𝑧) ∈ℝ<sup>3</sup> |2𝑑≤𝑧< 2𝑑+ 2}. Since regions corresponding to different sub-spaces do not overlap, interference between them is avoided. Details regarding the ray launching positions are provided in the following section. In addition to the position, the radius of spheres must also be determined, as mentioned before, we support dynamic radius for flexible trade-off. However, naively implementation necessitate runtime scene reconstruction which is very expensive. To mitigate the overhead, we propose a ray-time based method to support dynamic radius, which will be detailed in next subsection, so here we set the radius to be a constant number. 

**_Online Searching_** _._ We outline the process of a single query search in the online phase, as detailed in Algorithm 2. This can be easily extended to handle multiple queries. Recall that we now have a traversable scene builded upon the trained codebook, and indices that maintain the mapping from codebook entries to search points. When a query arrive, we first conduct the initialize filtering 

**ALGORITHM 1:** Build a traversable scene, prepare cluster centroids of filter and entry-search <u>points mapping offline.</u> 

|**Input**<br>**Outpu**|**:**𝑝𝑜𝑖𝑛𝑡𝑠[𝑁][𝐷],𝑀= 2,𝐸,𝐶,𝑚𝑒𝑡𝑟𝑖𝑐<br>**t:**𝑀𝑎𝑝[𝐶][ <sup>𝐷</sup><br>𝑀<sup>]{𝑒𝑛𝑡𝑟𝑦_𝑖𝑑∶𝑝𝑜𝑖𝑛𝑡𝑠_𝑖𝑑[ ]}, 𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑𝑠, 𝑙𝑎𝑏𝑒𝑙𝑠, 𝑆𝑐𝑒𝑛𝑒</sup>|
|---|---|
|1: **fu**<br>2:<br>3:|**nction**BuildRTScene(𝑝𝑜𝑖𝑛𝑡𝑠[𝑁][𝐷],𝑀,𝐸,𝐶)<br>𝑆𝑐𝑒𝑛𝑒, 𝑓𝑖𝑙𝑡𝑒𝑟, 𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑𝑠, 𝑙𝑎𝑏𝑒𝑙𝑠←∅, 𝑘𝑚𝑒𝑎𝑛𝑠(𝑝𝑜𝑖𝑛𝑡𝑠, 𝑛_𝑐𝑙𝑢𝑠𝑡𝑒𝑟= 𝐶), 𝑓𝑖𝑙𝑡𝑒𝑟.𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑𝑠, 𝑓𝑖𝑙𝑡𝑒𝑟.𝑙𝑎𝑏𝑒𝑙𝑠<br>𝑟𝑒𝑠𝑖𝑑𝑢𝑎𝑙←[𝑥−𝑐𝑒𝑛𝑡𝑜𝑖𝑑𝑠[𝑥.𝑙𝑎𝑏𝑒𝑙]𝑓𝑜𝑟𝑥𝑖𝑛𝑝𝑜𝑖𝑛𝑡𝑠]<br>|
|4:|**for**𝑠∈[0, <sup>𝐷</sup><br>𝑀<sup>)</sup><sup>**do**</sup>|
|5:|𝑟𝑒𝑠, 𝑐𝑜𝑑𝑒𝑏𝑜𝑜𝑘, 𝑒𝑛𝑡𝑟𝑖𝑒𝑠←𝑟𝑒𝑠𝑖𝑑𝑢𝑎𝑙[∶, 2𝑠∶2𝑠+ 2], 𝑘𝑚𝑒𝑎𝑛𝑠(𝑟𝑒𝑠, 𝑛_𝑐𝑙𝑢𝑠𝑡𝑒𝑟= 𝐸), 𝑐𝑜𝑑𝑒𝑏𝑜𝑜𝑘[𝑠].𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑𝑠|
|6:|**for**𝑒∈[0, 𝐸)**do**|
|7:|𝑥, 𝑦, 𝑧←𝑒𝑛𝑡𝑟𝑖𝑒𝑠[𝑒].𝑥, 𝑒𝑛𝑡𝑟𝑖𝑒𝑠[𝑒].𝑦, 2𝑠+ 1; 𝑆𝑐𝑒𝑛𝑒.𝑎𝑑𝑑(𝑠𝑝ℎ𝑒𝑟𝑒(𝑝𝑜𝑠= (𝑥, 𝑦, 𝑧), 𝑟= 𝐶𝑜𝑛𝑠𝑡))|
|8:|**for**𝑒∈[0, 𝐸), 𝑐∈[0, 𝐶)**do**|
|9:|𝑟𝑒𝑠𝑐, 𝑀𝑎𝑝[𝑐][𝑒] ←[𝑙𝑎𝑏𝑒𝑙𝑠[𝑝] = 𝑐𝑓𝑜𝑟𝑝𝑖𝑛𝑟𝑒𝑠], [_p encoded by e for p in_𝑟𝑒𝑠𝑐]|
|10:|**return**𝑆𝑐𝑒𝑛𝑒, 𝑀𝑎𝑝, 𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑𝑠, 𝑙𝑎𝑏𝑒𝑙𝑠|



ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:10 

to select several (𝑛𝑝𝑟𝑜𝑏𝑠) IVF clusters, and following calculation only happen in search points belong to these clusters. Next, we compute the residuals between query and centriods of chosen IVF clusters, since codebooks are trained with residuals. Then, we use the trained regression model with a user defined scaling factor to get a dynamic radius, and convert this radius to the ray’s maximum travel time (the details will be discussed in next subsection) (Line 3–6). Then, we create a ray originating from the coordinate of the residual (between query projection and IVF cluster centroid projection) in every sub-spaces. For 𝑧-coordinate, we set the origin of the ray to 𝑧= 2𝑑 for 𝑑<sup>𝑡ℎ</sup> sub-space, ensuring that the ray interact only with the spheres in the same sub-space. 

Line 9 register a callback, RT_HitShader, which is triggered when a ray intersects a sphere, i.e., when an entry lies closely with the query projection in a sub-space. Within this hit callback, we compute and store the actual distance via ray time (detailed in next subsection) (Line 10–11). To handle multiple queries, we launch rays fro all projections concurrently, keeping a separate list for each query to record the hit entries and their distance, These lists together constitute the distance look-up table (Line 12–13). With the look-up table, we accumulate the distance for search points via querying corresponding item in entry-point indices in every sub-spaces. The item represents the distance between query projection and codebook entry in a sub-space. Finally, the distance from all sub-spaces are accumulated and returned for sorting and top-k process (Line 7–8). 

## **4.3 Optimizations** 

Aforementioned contents describe how to leverage the sparsity in PQ and how to utilize RT core to enable efficient pruning and calculation. While we still need several optimization to mitigate runtime overhead, fully utilize the hardware and seek for higher search performance. 

**_Fast Distance Calculation and Dynamic Radius with Ray Time_** _._ There are two crucial concerns in aforementioned process, the first is the calculation of hit distance, which represent the true distance between queries and codebook entries. And the second is dynamic radius to enable flexible trade-off. The former may introduce tremendous sphere attribute querying and the latter may introduce expensive scene reconstruction. Fortunately, both can be solved with the concept of ray traveling time. 

There are two time related to the ray in RT pipeline: 𝑡ℎ𝑖𝑡 and 𝑡𝑚𝑎𝑥, the first represent the elapsed time between ray is launched and ray hits an object, and the second represent the maximal time a 

## **ALGORITHM 2:** Construct L2-LUT with the RT core and conduct distance calculation for interested search points. 

|**Inpu**|**t:**𝑞𝑢𝑒𝑟𝑖𝑒𝑠[𝑄][𝐷],𝑖𝑛𝑑𝑒𝑥,𝑞𝑢𝑒𝑟𝑦_𝑠𝑒𝑙𝑒𝑐𝑡_𝑐𝑙𝑢𝑠𝑡𝑒𝑟𝑠,𝑛𝑝𝑟𝑜𝑏𝑠,𝑑𝑒𝑛𝑠𝑖𝑡𝑦_𝑚𝑎𝑝,𝑝𝑜𝑙𝑦_𝑟𝑒𝑔𝑟𝑒𝑠𝑠𝑜𝑟,𝑡ℎ𝑟𝑒𝑠_𝑠𝑐𝑎𝑙𝑒(user defined)|
|---|---|
|**Outp**|**ut:**𝐿2_𝐿𝑈𝑇[𝑄][𝑛𝑝𝑟𝑜𝑏𝑠][ <sup>𝐷</sup><br>𝑀<sup>]{𝑒𝑛𝑡𝑟𝑦_𝑖𝑑∶𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒}</sup>|
|1: **f**|**unction**L2_LUT(𝑞𝑢𝑒𝑟𝑦,𝑖𝑛𝑑𝑒𝑥,𝑞𝑢𝑒𝑟𝑦_𝑠𝑒𝑙𝑒𝑐𝑡_𝑐𝑙𝑢𝑠𝑡𝑒𝑟𝑠)<br>|
|2:|**for**𝑞∈[0, 𝑄), 𝑠∈[0, <sup>𝐷</sup><br>𝑀<sup>)</sup><sup>**do**</sup>|
|3:|𝑥, 𝑦, 𝑧←𝑞[_𝑞][2𝑠∶2𝑠+ 1], 2𝑠|
|4:|**for**𝑐**in**𝑞𝑢𝑒𝑟𝑦_𝑠𝑒𝑙𝑒𝑐𝑡_𝑐𝑙𝑢𝑠𝑡𝑒𝑟𝑠[𝑞]**do**|
|5:|𝑡ℎ𝑟𝑒𝑠, 𝑡←𝑝𝑜𝑙𝑦_𝑟𝑒𝑔𝑟𝑒𝑠𝑠𝑜𝑟(𝑑𝑒𝑛𝑠𝑖𝑡𝑦_𝑚𝑎𝑝(𝑥, 𝑦)), 1 −√<br>1.0<sup>2 </sup>−(𝑡ℎ𝑟𝑒𝑠× 𝑡ℎ𝑟𝑒𝑠_𝑠𝑐𝑎𝑙𝑒)<sup>2</sup>|
|6:|<br>𝑥, 𝑦←(𝑥, 𝑦) −𝑖𝑛𝑑𝑒𝑥.𝑐𝑒𝑛𝑡𝑟𝑜𝑖𝑑𝑠[𝑐][2𝑠∶2𝑠+ 1];𝑟𝑎𝑦𝑠.𝑎𝑑𝑑(𝑥, 𝑦, 𝑧, 𝑡𝑚𝑎𝑥= 𝑡, 𝑑𝑖𝑟= (0, 0, 1))|
|7:|𝑖𝑛𝑑𝑒𝑥.𝑠𝑐𝑒𝑛𝑒.𝑠𝑒𝑡_ℎ𝑖𝑡_𝑐𝑎𝑙𝑙𝑏𝑎𝑐𝑘(**RT_HitShader**)|
|8:|**return RayTracing(**𝑟𝑎𝑦𝑠, 𝑠𝑐𝑒𝑛𝑒**)**|
|9: **f**|**unction**RT_HitShader(𝑖𝑛𝑑𝑒𝑥,𝑞𝑢𝑒𝑟𝑦_𝑠𝑒𝑙𝑒𝑐𝑡_𝑐𝑙𝑢𝑠𝑡𝑒𝑟𝑠)|
|10:|𝑟𝑎𝑦, 𝑠𝑝ℎ𝑒𝑟𝑒, 𝑡ℎ𝑖𝑡, 𝑞, 𝑠, 𝑒, 𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒←**GetRay(),GetHitSphere(),GetTime()**|
|11:|𝑞, 𝑠, 𝑒, 𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒←𝑟𝑎𝑦.𝑞𝑢𝑒𝑟𝑦_𝑖𝑑, 𝑟𝑎𝑦.𝑠𝑢𝑏𝑠𝑝𝑎𝑐𝑒_𝑖𝑑, 𝑠𝑝ℎ𝑒𝑟𝑒.𝑒𝑛𝑡𝑟𝑦_𝑖𝑑,√<br>𝑅<sup>2 </sup>−(1 −𝑡ℎ𝑖𝑡)<sup>2</sup>|
|12:|**for**𝑐**in**𝑞𝑢𝑒𝑟𝑦_𝑠𝑒𝑙𝑒𝑐𝑡_𝑐𝑙𝑢𝑠𝑡𝑒𝑟𝑠**do**|
|13:|𝐿2_𝐿𝑈𝑇[𝑞][𝑐][𝑠].𝑎𝑑𝑑({𝑒∶𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒})|



ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:11 



<!-- Start of picture text -->
(b) (c)<br>Z = 2  ⇥  s Ray Origin Solo-run 64<br>d=p R 2 − (1 − thit ) 2 eqQDA=</latxi> t1- maxpR = 2 − r 2 3.02.0 Co-runTensor core impl.Solo-run, 10%Co-run   , 10% 4832 Hit Count Top-10% +++ +++ +++<br>d r 1.0 16 1 0 1 0-1<br>R R Hit countw/ panelty<br>0.0 0<br>Z = 2  ⇥  s + 1 Unreachable max t max = 1 Distance LUTConstruction AccumulationDistance Top 0.1% True L2 distanceTop 1% Top 10% Top 100%<br>t hit miss hit Hit count<br>L2<br>Normalized latency Top-10%<br><!-- End of picture text -->

Fig. 5. (a) We use the 𝑡ℎ𝑖𝑡 to calculate the hit distance between query projection and sphere centroid. For dynamic radius, we use 𝑡𝑚𝑎𝑥 to adjust the area can be hit by rays. (b) Latency breakdown of distance look up table construction and distance accumulation, **solo-run** represents the latency of letting RT exclusively own the hardware to run and **co-run** represents naive co-execution of RT and distance accumulation on the hardware, and **10%** represents a 9:1 resource partition via CUDA MPS. (c) Relationship between hit count and exact distance. 

ray can travel before it vanishes. We use the first to calculate exact hit distance and the second to support dynamic radius, as shown in Figure 5. Recall that we set the radius of all spheres to be a constant number. To be specific, different hit distance will lead to different 𝑡ℎ𝑖𝑡. As shown in the left part of Figure 5, the hit distance 𝑑 can be calculated with the constant radius and the 𝑡ℎ𝑖𝑡, a longer 𝑡ℎ𝑖𝑡 implies a larger hit distance. Note that the hit time is stored in the register of a ray thread. So, we don’t need to query the position and radius of the hit sphere in the slow off-chip memory to calculate the hit distance, instead we access the register data and conduct several simple arithmetic operations. For dynamic radius, we adjust the 𝑡𝑚𝑎𝑥 to enable. As shown in the right part of Figure 5, a ray with half of the life-span can only hit a smaller region comparing to the original sphere. So, if we need a smaller dynamic radius, we pass a smaller 𝑡𝑚𝑎𝑥 when we launch rays, instead of adjust the spheres’ radius and reconstruct the scene. For instance, if user provide a scaling factor of 0.8 on a 0.6 original radius, the 𝑡𝑚𝑎𝑥 = 0.64. 

**_Pipelining on Heterogeneous Cores_** _._ With aforementioned RT core implementation, part of the calculation is offloaded from CUDA core to the RT core. However, notice that the RT is only responsible for the quick intersection test, the shader code is still executed on the CUDA core (RT core only determine whether or not to execute RT_HitShader, while all RT_HitShader are executed on the CUDA core). Given distance accumulation calls CUDA core too, naive co-execution can lead to significant interference and performance degradation, as shown in Figure 5(b). To solve, we offload the distance accumulation to Tensor core. The distance of selected search points in sub-spaces are arranged into rows to form matrix A, with dimensions (𝑀, 𝐾) = (𝑄×sizeof(selected points)×IVF clusters,𝐷/2). Next, matrix B is constructed with (𝐾, 𝑁) = (𝐷/𝑀, 1), with all elements equal to 1.0.The accumulation is performed via matmul(𝐴, 𝐵) using cublas library. Moreover, we leverage CUDA MPS to allocate SM resources in a 9 ∶1 ratio. This results in comparable latencies of distance look-up table construction and distance accumulation. Finally, the data rearrangement brings less than 5% overhead in latency, also shown in Figure 5(b). 

**_Aggressive Approximation: Hit Count-based Method_** _._ Though proposed algorithm and RT core implementation save a lot of calculation, there are still multiple floating point operation to compute the hit distance. We propose a more aggressive approximation that only leverages the hit/miss result from the RT core, drawing inspiration from previous work [34]. We first investigate the correlation between hit count and exact distance, where all spheres are assigned a radius that includes the top-100 neighbors. As shown in Figure 5(c), these two factors are strongly correlated. The reason is that a higher hit count indicates proximity to the query projection acroess multiple sub-spaces. This observation motivates the development of a hit count-based method. 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:12 

We utilize a reward/penalty-based morel, that creates an additional sphere with half the radius of each original sphere, as illustrated in Figure 5(c). The hit count is incremented by one when the ray intersects the inner sphere and decremented by one as a penalty when the ray misses both spheres. As shown in Figure 5(c), the hit count (blue triangle) derived from this method demonstrates a stronger corelation than the original hit count (yellow hexagon). While this approximation may lead to false positives and negatives (- and + in Figure 5(c)), it provides a new parameter to balance search quality and performance. 

## **5 Accelerating Sparse Attention via Proposed Ray Tracing-based ANNS** 

In this section, we propose accelerating the retrieval-based sparse attention mechanism in LLMs using our sparsity-aware ANNS algorithm and corresponding RT core mapping. We first explain the concept of retrieval-based sparse attention, then outline the required modifications to apply our RT approach for acceleration. Finally, we present the overall workflow and user interface for system-level integration. 

## **5.1 Basic Idea** 

As the context length (i.e., number of tokens) that LLMs must process increases, the attention mechanism accounts for a growing proportion of inference and serving time. This is because the computation required by attention grows linearly with the number of generated tokens due to key and value caching. Our profiling using SGLang [71] shows that attention accounts for 88% of the execution time when decoding a single token across 16 requests with a 16k context length using LLaMA-7B–a context length, that is, not particularly large. Therefore, optimizing attention computation is critical for improving inference speed. 

Recent studies suggest that not all attention scores are necessary for computation. RetrievalAttention [36] shows that retaining only the top 1%–3% of attention scores yields accuracy nearly identical to using the full set. Their method selects the top 1%–3% closest keys from the key cache, performs a partial softmax to obtain attention logits, and multiplies them with the corresponding values to produce the final attention output. This sparse attention mechanism can be represented as follows: 



In this process, 𝑇𝑜𝑝𝐾 is typically implemented using the FAISS library, either on CPU or GPU. Even with the GPU version, it introduces considerable overhead due to additional kernel launches and inefficient implementation. The core idea of JUNO++ is to accelerate attention by adopting the RetrievalAttention algorithm while replacing the original 𝑇𝑜𝑝𝐾 component with our RT version. To enable this, several workflow adjustments are required. 

## **5.2 Ray Tracing Core-based Implementation** 

Integrating RT acceleration into retrieval-based sparse attention introduces several challenges to the original workflow. The first challenge is managing the growing set of search points, i.e., keys, as frequent scene reconstruction occurs when incrementally building the RT scene during LLM inference. The second challenge is efficiently supporting the inner-product metric. The proposed RT implementation for ANNS is designed for L2 distance, while similarity in LLMs is typically measured by the inner product between queries and keys. Existing ANNS methods that support inner products either introduce misaligned extra dimensions or are restricted to cosine similarity, a special case of the inner product. These challenges must be addressed for RT to effectively accelerate retrieval-based sparse attention. 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:13 

To address the first challenge, we introduce a mechanism that freezes the codebook–and thus the RT scene–while updating the inverted-file index at runtime. For the second challenge, we propose a cost-free, radius-based transformation to support inner-product similarity without incurring runtime overhead. 

**_Preparing Codebook and Inverted-File Index of Keys_** _._ Building the RT scene with all keys causes frequent scene reconstruction at runtime due to the auto-regressive nature of LLMs, which is impractical. Instead, we propose constructing the RT scene using codebooks derived from the keys and maintaining inverted-file indices from codebook entries to keys at runtime, as the latter introduces significantly less overhead than scene reconstruction. Specifically, we apply PQ to the keys after they are generated during the prefilling stage to obtain a set of codebooks, which remain fixed throughout the generation process. During decoding, each newly generated key updates the inverted-file indices by identifying the closest codebook entries in each subspace and appending its ID to the corresponding index. Notably, we do not update the codebook entries–or the RT scene–during decoding. After updating the inverted-file indices, we retrieve the top-k closest key indices for a given query by: (i) locating the nearest codebook entries in each subspace using RT, and (ii) retrieving the corresponding key IDs via the inverted-file indices, followed by computation with the retrieved indices. The PQ configuration uses 256 codebook entries per two-dimensional subspace, consistent with our sparsity-aware ANNS algorithm. 

**_Inner-Product Metric Support_** _._ Our method enables efficient computation of **maximum inner product similarity** ( **MIPS** ) with minimal overhead thanks to a well-crafted transformation mechanism. In contrast to previous approaches, which add extra dimensions to minimize the L2 distance between transformed queries and search points to maximize the inner-product, we propose to support the metric with no extra dimension since they may lead to un-aligned computation and storage. Recalling that we use ray traveling time 𝑡ℎ𝑖𝑡 to calculate L2 distance as follows: 



Inner-product is implicitly calculated in L2 calculating, since inner-product can be calculated as follows: 



The coordinates of codebook entry, 𝑥𝑒𝑛𝑡𝑟𝑦, 𝑦𝑒𝑛𝑡𝑟𝑦, typically require global memory access at runtime to compute the inner product 𝐼𝑃(𝑒𝑛𝑡𝑟𝑦, 𝑞𝑢𝑒𝑟𝑦) from 𝐿2(𝑒𝑛𝑡𝑟𝑦, 𝑞𝑢𝑒𝑟𝑦). However, with RT core, we can substitute original radius 𝑅𝑠𝑝ℎ𝑒𝑟𝑒 with 𝑅𝑠𝑝ℎ𝑒𝑟𝑒′<sup>=</sup> <u>√𝑅𝑠ℎ𝑝𝑒𝑟𝑒</u><sup>2+ 𝑥2𝑒𝑛𝑡𝑟𝑦+ 𝑦2𝑒𝑛𝑡𝑟𝑦, eliminating</sup> the need for extra dimensions, as shown below: 



Here, 𝑥𝑞𝑢𝑒𝑟𝑦, 𝑦𝑞𝑢𝑒𝑟𝑦, 𝑅𝑠𝑝ℎ𝑒𝑟𝑒 are constant values. Therefore, the inner product can be computed directly with new hit time, eliminating the need for sphere coordinate access. Moreover, the coordinates of query can be ignored as it is constant across all entries. As a result, only the radius 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:14 







Fig. 6. Negative MIPS (left) may lead to large L2 distance (middle) with our conversion, which introduces potential false positive. (right) We utilize visibility mask to select proper parts of 𝑥𝑂𝑦 to check intersection results to filter out false negatives. 

of spheres needs to be adjusted from original radius for L2 distance to the new one during offline processing. Almost no runtime overhead is introduced. 

A remaining challenge is that converting a metric with both meaningful positive and negative values (inner product) to one with only positive values (L2 distance) may lead to false positives, as illustrated in Figure 6(left). In the left example, when the red query arrives, the blue point should be ignored because it yields a negative inner product with the query. However, after the conversion, the query ray still hits the blue point and returns it–even with a relatively large distance–as shown on the right. Since MIPS is a larger-is-better metric, the blue point is mistakenly returned as a potential neighbor. Fortunately, OptiX provides an 8-bit visibility mask [46], which enables efficient selective visibility for different objects in the scene. Using this mask, we divide the 𝑥𝑂𝑦 plane into eight regions. When a query arrives, we configure the ray’s visibility mask to select the appropriate area for intersection testing. A parameter, 𝑁𝑃𝑎𝑟𝑡𝑠, controls how many regions are tested for intersection, allowing a trade-off between accuracy and performance. In our implementation, we empirically set 𝑁𝑃𝑎𝑟𝑡𝑠 = 2, meaning we consider 5/8 of the 𝑥𝑂𝑦 plane. This spatial partitioning and filtering technique improves both quality and performance by reducing the number of spheres checked for intersection. 

## **5.3 System Integration** 

After addressing the challenges associated with inner-product-based attention computation, we present how our optimizations are integrated into existing systems to enable fast retrieval-based sparse attention. We first describe the provided interfaces, followed by an overview of the overall workflow utilizing these interfaces. 

**_User Interface_** _._ We provide users with several interfaces, ranging from preparing codebooks during the prefilling stage to calculating top-k attention scores at the decoding stage. These interfaces are listed below with detailed descriptions: 

- codebook, ivf = Train_and_Build_Index(keys, configs). This interface returns (i) a trained codebook based on the input data (key cache) and user-defined configurations, and (ii) indices mapping codebook entries to search points. Users must call this interface during the prefilling stage, providing the keys computed from input prompt tokens and the key projection weights. For configs, the PQ dimension is fixed at two, while users can specify the number of cluster centroids to train and whether to share the codebook across batch or head dimensions. After training, this interface invokes OptiX APIs to create a RT scene stored in GPU memory. 

- configs = “attr”: “value”. This interface is used to pass various optimization switches, including aggressive hit-count-based approximation and region-based inner-product filtering. Usage follows the format optimization: on/off. 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:15 





Fig. 7. (left) Prefiling stage of new workflow, we use the interface to train codebook and build RT core compitable scene. (right) At decoding stage, we directly use ray traced approximate distance to conduct sparse attention calculation. 

- indices = Locate(query, codebook, ivf, and topk). This interface initiates a call to the RT core-based ANNS process. Using the pre-built scene, we (i) launch rays from the positions of query projections in each sub-space, (ii) use the ivf to perform approximate distance accumulation following the JUNO++ algorithm, and (iii) apply bitonic sort to rank distances and obtain the final top-k indices. Depending on user requirements, this interface returns the indices of the top-k closest keys. We evaluate speedups with varying retained attention scores by adjusting ANNS recall in the next section. 

- output = QK_Softmax(query, keys, values, and indices). This interface consists of two parts: QK and Softmax. Users employ this interface to compute the inner product between queries and keys and proceed with subsequent calculations. Notably, JUNO++ is used only to retrieve the closest key indices (Locate), while attention is still computed precisely using the corresponding keys and values. The speedup arises because fewer keys and values need to be loaded. This calculation is consistent with RetrievalAttention, as is its accuracy. 

- Update(key, codebook, and ivf). This interface processes a newly generated key by inserting it into the ivf through locating its closest entry in the codebook for each sub-space. 

**_Overall Workflow_** _._ Finally, we present the overall workflow using our interfaces, as illustrated in Figure 7, with the prefiling stage on the left and the decoding stage on the right. Similar to our optimized ANNS pipeline, the linear projection and RT components in sparse attention computation are fully pipelined: the RT core begins processing as soon as a new key is generated and the previous key cache is loaded, while value projection is simultaneously computed by the CUDA core. The primary speedup of our pipeline comes from eliminating the time-consuming 𝑞× 𝑘<sup>⊤</sup> calculation by offloading it to the highly efficient RT core. However, the RT pipeline’s launch time is relatively long compared to inference latency in short-context scenarios, as demonstrated in later evaluations. Therefore, our solution is best suited for long-context scenarios. 

## **6 Evaulation** 

We validate the effectiveness of the proposed algorithm and hardware mapping in JUNO++ through comprehensive experiments. 

## **6.1 Experimental Setup** 

**_Setup._** The RT component of JUNO++ is developed using NVIDIA OptiX 7.6 [46]. We evaluate JUNO++ on various NVIDIA GPUs. The specifications of these GPUs are listed in Table 1. Note that 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:16 

Table 1. Specifications of Evaluated GPUs 

|Item|RTX 4090|Tesla A40|Tesla A100|
|---|---|---|---|
|Architecture|Ada (2022)|Ampere (2020)|Ampere (2020)|
|CUDA Core #|16,384|10,752|6,912|
|Tensor Core #|512 (4th Gen)|336 (3th Gen)|432 (3th Gen)|
|RT Core #|128 (3th Gen)|84 (2nd Gen)|0|
|DRAM|24 GB(GDDR6X)|48 GB(GDDR6)|40 GB(HBM2)|



OptiX falls back to CUDA cores for RT operations on GPUs lacking RT cores, enabling JUNO++ to remain compatible with such hardware. This allows us to perform a sensitivity analysis to examine (i) the benefits of algorithmic improvements alone, and (ii) the influence of RT hardware performance. Finally, we assess the performance of RT based sparse attention with Llama series configuration [61] where the head dimension is 128. 

**_Dataset._** This work targets enhancing the efficiency of high-dimensional ANN search on a single GPU. We evaluate our approach using widely adopted datasets: SIFT1M, SIFT100M [30], DEEP1M, DEEP100M [5], and TTI1M [58], where 1M and 100M denote 1 million and 100 million data points, respectively. The embedding dimensions are 128/128, 96/96, and 200, accordingly. While TTI1M employs inner product search (MIPS), the others use L2 distance as the similarity metric. It is important to note that datasets larger than 1 billion entries exceed the memory capacity of a single GPU and thus require techniques such as chunking, partitioning, or other storage-efficient methods proposed in complementary works [10, 53, 60]. For instance, GGNN and ANNA leverage 8 and 12 accelerators, respectively, to handle billion-scale datasets [21, 35]. 

**_Baseline and Configurations._** We primarily benchmark JUNO++ against FAISS, a leading GPU-based library for ANN search [31]. To fairly assess our method, we configure JUNO++ with the same number of IVF clusters as FAISS and perform evaluations under multiple PQ settings. We further extend our evaluation by incorporating the commonly used HNSW optimization [41] alongside IVF and PQ. Note that HNSW is an orthogonal indexing technique compatible with IVF. Since HNSW still involves distance computations and neighbor sorting during search, it can benefit from the optimized PQ mechanism in JUNO++. While JUNO++ focuses on enhancing the PQ stage, full integration of HNSW into our IVF structure is left for future exploration. Nonetheless, we compare our design with FAISS baselines that apply HNSW, implemented using index_factory with the format IVFx_HNSWy,PQz. Additionally, existing CPU-based [24], RAM-based [10], and disk-based [10, 60] ANN methods are orthogonal and complementary to our approach. To evaluate the latency of RT based sparse attention, we compare the latency of Locate plus QK part with the original QK implemented with torch.matmul(). For RT version, QK after Locate receive much less data as input. Since RetrievalAttention has already verified that saved attention can significantly accelerate the Softmax part, we focus on the former part they do not cover. We mainly evaluate the speedup under long context scenario: 1M tokens due to high launch overhead of RT pipeline, which will be analyzed in later contents. 

**_Metric._** n our evaluation, we measure search quality using two metrics: Recall-1@100 (R1@100) and Recall-100@1000 (R100@1000). R1@100 is defined as the proportion of queries whose top-100 retrieved neighbors contain the ground-truth nearest neighbor. The ranking within the top-100 is not considered. For example, if 8 out of 10 queries retrieve the true nearest neighbor within their top-100 results, the R1@100 score would be 0.8. R100@1000 quantifies the average number of true top-100 nearest neighbors found among the top-1000 retrieved results. 

**_Evaluation Plan._** In this work, we evaluate the **Query Per Second** ( **QPS** ) and search accuracy of JUNO++ under different configurations. For each setup, a scaling factor is applied to balance performance and retrieval quality. 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:17 



<!-- Start of picture text -->
10 7 SIFT1M-R1@100 10 7 DEEP1M-R1@100 TTI1M-R1@100 10 6 SIFT100M-R1@100<br>10 6<br>10 6 10 6 IP Fixed 10 5<br>10 5 10 5 10 5 PQ40 JUNO-L<br>PQ64PQ32 PQ8+HNSW JUNO-MJUNO-L PQ48PQ24 PQ8+HNSW JUNO-MJUNO-L PQ20+HNSW JUNO 10 4 PQ(Best)+HNSW JUNOJUNO-L<br>10 4 PQ16 JUNO-H JUNO 10 4 PQ12 JUNO-H JUNO 10 4 JUNO-H JUNO-H<br>10 7 SIFT1M-R100@1000 10 7 DEEP1M-R100@1000 TTI1M-R100@1000 DEEP100M-R1@100<br>10 6 10 6<br>IP Fixed<br>10 6 10 6<br>10 5<br>10 5<br>10 5 10 5<br>10 4<br>10 4 10 4 10 3 10 4<br>0.4 0.6 0.8 1.0 0.4 0.6 0.8 1.0 0.4 0.6 0.8 1.0 0.4 0.6 0.8 1.0<br>Recall Recall Recall Recall<br>Query per second (QPS)<br><!-- End of picture text -->

Fig. 8. Result of QPS and search quality of JUNO++ on various datasets including SIFT1M, DEEP1M, TTI1M, SIFT100M, and DEEP100M. The bolded grey line labeled JUNO++ is the Pareto frontier of our search engine under different configurations (i.e., the configuration of JUNO++-L, JUNO++-M, and JUNO++-H), standing for the optimal performance of JUNO++ at a given search quality requirement. 

- JUNO++-H: We employ hit time based exact hit distance calculation for high quality. 

- JUNO++-M: We employ finer-grained hit count-based selection with multiple spheres for medium quality. 

- JUNO++-L: We employ hit count-based selection only for low quality. 

It is important to highlight that the search quality of different configurations may overlap. Based on empirical observations, we categorize the quality requirements into three ranges: [0.0, 0.95], [0.95, 0.97], and [0.97, 1.0]. Correspondingly, we refer to the configurations as JUNO++-L, JUNO++M, and JUNO++-H, respectively. If JUNO++-L or JUNO++-M does not satisfy the expected quality range, we default to JUNO++-H. 

We next analyze the performance gains contributed by the two optimization techniques. Finally, we perform a sensitivity study to assess the impact of key design choices in JUNO++. 

## **6.2 Search Quality and Throughput** 

Figure 8 presents the overall trade-off between search quality and throughput across different datasets. We consolidate the results from multiple configurations of JUNO++-L/M/H into the bold gray curve, as JUNO++ enables flexible adjustment between accuracy and performance. The curves labeled PQx in the FAISS baseline correspond to partitioning the vector space into x subspaces. The curves marked with +HNSW indicate the performance when the HNSW optimization is applied to the best-performing PQ setup, using its optimal configuration parameters. 

**_Justifications of Baseline Configurations._** We perform an in-depth analysis of the performance of various baseline methods and provide justifications for their respective configurations. Upon examining the effect of HNSW optimization, we find that it offers minimal improvement on smaller datasets (1M), while significantly enhancing performance on larger datasets (100M). These findings are consistent with the benchmark results presented in FAISS [43]. Regarding the PQ configuration, we observe that using more smaller subspaces improves search accuracy, but at the cost of throughput. We further optimize the baseline across different datasets, driving its performance closer to the Pareto frontier. Despite the varied results of the baseline methods, we will compare JUNO++ with the top-performing baseline in our subsequent analysis to ensure a fair and thorough evaluation. 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:18 

**_SIFT1M and DEEP1M._** We achieve a 7.8× improvement in QPS compared to the baseline under the low search quality condition: JUNO++-L with R1@100≤0.95. This performance gain arises from leveraging sparsity and aggressive approximation. First, by applying a stricter threshold, we are able to eliminate more search points, enhancing filtering efficiency. Second, we use hit count-based selection, avoiding the need for actual distance calculations. Moreover, the tighter threshold reduces the size of the spheres, leading to fewer hit events. As a result, JUNO++ can fully leverage the sparsity and spatial locality of the data, providing a significant advantage for low search quality scenarios. 

The high search quality requirement, such as R1@100=0.99, leads to an increased number of selected search points. Moreover, calculating the exact distances is essential to satisfy this strict accuracy constraint. These factors reduce the benefits of exploiting sparsity. Nonetheless, JUNO++ still delivers a 2.4× throughput improvement over the baseline, thanks to the tree-based search approach used in the RT core. 

It is important to note that JUNO++-L reaches only 0.95 recall for these two datasets due to its reliance on a pure hit count-based approximation approach. JUNO++-M enhances the search quality to 0.97 by using the reward/penalty-based approximation with additional inner spheres. The throughput increases by 2.9× compared to the baseline. 

**_TTI1M._** This dataset employs the inner product metric (MIPS). Since JUNO++-H computes the exact distance within each subspace, its performance improves by 2.04×, similar to the previous datasets with the L2 metric. It is worth noting that the FAISS baseline also achieves a recall of 0.96. However, the hit count-based method disregards the 𝑡ℎ𝑖𝑡 information, where intersection only indicates proximity in terms of L2 distance, rather than similarity in the context of the inner product. As a result, search quality degrades quickly when using the inner product metric, causing the JUNO++-L line to shift left. Additionally, we present the result of our improved version, utilizing space partitioning and filtering techniques, which achieves the highest recall in JUNO++-H. This leads to a 1.3× increase in QPS, with a slight improvement in recall about 2% in R1@100, indicating the necessity of space partition and filtering mechanism. 

**_SIFT100M and DEEP100M._** The JUNO++-H and JUNO++-L configurations show average improvements of 1.5× and 2.1× over the baseline, respectively. The performance boost of JUNO++-H is limited as the _distance calculation_ becomes the primary bottleneck. It is important to note that these improvements are based on comparing JUNO++ **without** HNSW to FAISS **with** HNSW optimization. We chose not to implement HNSW in JUNO++ due to its high code complexity within the current FAISS framework. Additionally, integrating HNSW would not contribute additional insights for optimizing PQ. Interestingly, JUNO++-H still achieves a 3.0× performance increase over the baseline without the HNSW optimization. 

**_Results of Different Metrics._** Figure 8 shows the R100@1000 results for SIFT1M, DEEP1M, and TTI1M. JUNO++ demonstrates similar improvements over the baseline, validating the effectiveness of the approximation methods even under more stringent metrics. On average, 65% of the true top 100 nearest neighbors are included among the top 100 retrieved from 1000. This performance is also impacted by the quality of clustering in IVF and the PQ approximation, both of which are consistent in JUNO++ compared to the baseline. Moreover, the fix mechanism introduced in sparse attention support can get both higher accuracy and higher performance, as shown in Figure 8 as red marks. 

**_Results of RT-based Attention._** As shown in Figure 9(a), under long-context scenarios (𝑆𝑒𝑞_𝐿𝑒𝑛≥1𝑀), the combined latency of Locate and QK is slightly lower than that of the original full-attention QK. Moreover, since the algorithm allows flexibility in retaining 1%–3% of attention scores, we can tolerate lower recall for reduced latency. At 90% recall (2.7% scores retained), latency is nearly halved (54%), and, as verified by RetrievalAttention [36], accuracy loss is minimal. Conversely, baseline latency for shorter contexts is much lower than the RT implementation, due to the 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:19 



<!-- Start of picture text -->
400 (a) 1000 (b)<br>Baseline q×k.T latency: 307 us 100<br>200 10<br>1<br>0<br>0.99 0.98 0.97 0.95 0.90 0.5<br>Top-3% Recall Top-2.7%<br>Fig. 9. (a) Latency of RT-based 𝑄× 𝐾 𝑄× 𝐾 ⊤ under different recall quality. (b) Latency of RT pipeline launch. (c) We<br>overlap the RT pipeline launching overhead with the bubble between LLM inference kernels.<br>(a) (b) (c) 10 (d)<br>43 FAISSJUNOw/o pipeline 10 6 10 6 Tesla A100Tesla A40RTX 4090<br>w/o hit count<br>5<br>2 10 5<br>R-Small<br>1 10 5 R-LargeR-Dynamic FAISSJUNO w/o RT Core<br>0 10 4 0<br>0.99 0.97 0.95 0.9 0.8 0.6 0.8 0.9 1.0 0.5 0.6 0.7 0.8 0.9 1.0 0.99 0.97 0.95 0.9 0.75 0.6<br>Recall (R1@100) Recall (R1@100) Recall (R1@100) Recall (R1@100)<br>Seq Len=4k Seq Len=16k Seq Len=512k Seq Len=1M LaunchRT<br>Latency (us)<br>Latency (us)<br>Speed-up ratio Speed-up ratio<br>Query per second (QPS) Query per second (QPS)<br><!-- End of picture text -->

Fig. 9. (a) Latency of RT-based 𝑄× 𝐾 𝑄× 𝐾<sup>⊤</sup> under different recall quality. (b) Latency of RT pipeline launch. (c) We overlap the RT pipeline launching overhead with the bubble between LLM inference kernels. 

Fig. 10. (a) Improvement breakdown of JUNO++ against FAISS. (b) Performance of different threshold strategy, evaluated on NVIDIA Tesla A40. (c) QPS and recall of JUNO++ and FAISS on A100. (d) Average advantage against FAISS on different GPUs. 

substantial overhead of launching the RT pipeline, as shown in Figure 9(b) (note that subsequent Softmax benefits from reduced calculation but is excluded here for fairness). To mitigate this overhead, we apply a pipelining mechanism. Our profiling with SGLang [71], illustrated in Figure 9(c), reveals a significant bubble between linear projection and attention calculation, varying by model and configuration. This bubble allows launching the RT pipeline to overlap most of its overhead once query projection is complete. However, on systems with CUDA Graph and aggressive fusion techniques such as torch.compile() [2], this bubble may be absent. Thus, our implementation is generally suitable for long-context scenarios, typically those exceeding 512k tokens. 

## **6.3 Effectiveness of Different Optimizations** 

We now assess the impact of the optimizations in JUNO++, including the pipelining between CUDA-tensor-RT cores, hit count-based L2-LUT selective construction, and dynamic radius (i.e., distance threshold). Figure 10(a) illustrates the overall improvement as well as the effects when excluding the first two optimizations, while (b) demonstrates the impact of the dynamic radius. 

**_Overall Improvement._** JUNO++ demonstrates an average QPS improvement ranging from 2.1× to 4.4× across five datasets, from high to low search quality requirements, compared to the baseline. Even without dataset-specific tuning, JUNO++ achieves a maximum improvement between 8.5× and 3.2× on these datasets. 

**_Effectiveness of Pipelining._** The third bar in Figure 10(a) illustrates the performance improvement without pipelining. In high search quality scenarios, the bottleneck is the _L2-LUT construction_ , which has higher latency than distance calculation. As a result, omitting pipelining leads to a 44% reduction in improvement. Conversely, in cases where lower search quality is acceptable, the latencies of _L2-LUT construction_ and _distance calculation_ are comparable, causing a 50% decrease in improvement when pipelining is not used. 

**_Effectiveness of Hit Count-based Selection._** The final bar in Figure 10(a) shows the performance improvement when hit count-based selection is not applied. For very high search quality requirements, hit count-based selection has no effect, as it cannot achieve such high quality. However, 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:20 

as the search quality requirement relaxes, the impact of hit count-based selection becomes more significant, as lower recall demands fewer exact distance calculations. In summary, a combination of hit count-based selection and precise distance computation is essential for maintaining high search throughput across various search quality levels. 

**_Effectiveness of Dynamic Threshold Strategy._** We assess the search quality and throughput (QPS) using both small and large static thresholds. This evaluation is performed on SIFT1M with JUNO++-H, where the static thresholds are derived from the minimum and maximum values of the dynamic threshold. As shown in Figure 10(b), a large static threshold decreases search throughput but improves search quality. This occurs because a larger threshold increases the sphere size, resulting in rays intersecting more spheres and triggering additional hit shader functions. On the other hand, a smaller static threshold enhances throughput but compromises search quality. Even with low search quality requirements, selecting more clusters to address recall issues diminishes the performance gain from fewer hit shader invocations. For higher search quality demands, the small static threshold fails by missing too many true top-k neighbors, thereby reducing recall. In contrast, our dynamic threshold strategy surpasses both static approaches in terms of search quality and throughput. 

**_Sensitivity to RT Core Performance._** Finally, we assess the performance of JUNO++ with and without RT core acceleration on different GPUs. Figure 10(c) presents the detailed performance of JUNO++ and the baseline on the Tesla A100, a GPU lacking RT cores. This evaluation is conducted on the SIFT1M dataset with the baseline configured as PQ16+HNSW (the highest-performing configuration). Our results indicate that JUNO++ achieves substantial improvement at lower search quality requirements on the Tesla A100, suggesting that the gains are primarily attributed to the threshold-based selective algorithm. This also confirms the validity of utilizing sparsity and spatial similarity in the standard IVFPQ process. For higher search quality demands, JUNO++ gradually underperforms the baseline as the overhead from simulating RT with CUDA cores outweighs the minor benefits from sparsity. The results in Figure 10(c) suggest that JUNO++’s performance is limited by the RT cores’ capabilities. Therefore, we anticipate performance gains with faster RT cores. According to NVIDIA’s white paper on the Ada architecture [50], the Gen.3 RT core in Ada GPUs offers 2× the throughput of the Gen.2 RT core in Ampere GPUs. As depicted in Figure 10(d), on average across three 1M datasets, the RTX4090 achieves a 1.5× higher improvement over the baseline compared to the Tesla A40. Notably, the throughput of the CUDA and Tensor cores in the RTX4090 is 1.4× that of the A40 per SM [48, 50]. 

## **7 Related Work** 

Given that an ANN algorithm involves both indexing and encoding, we compare JUNO++ with existing methods in these two algorithmic areas, as well as in terms of hardware acceleration. 

_Indexing._ Indexing methods reduce the search space by eliminating irrelevant points. A common technique is the inverted file index (IVF)[37], which organizes data into clusters and selects the closest clusters for search. Graph-based approaches, like nearest neighbor graphs, also accelerate searches by pruning irrelevant data[15, 26, 64]. Heuristic methods [6, 17, 27] enhance these techniques. Notably, hierarchical navigable small world (HNSW)[41] and navigating spread-out graph (NSG)[18] are widely used. HNSW builds a hierarchical graph where the search explores deeper, higher-degree, and shorter edges, allowing efficient results. NSG further reduces graph size and search length by introducing navigation points. Additionally, tree-based methods such as kd-tree, octree [9, 44], and locality-sensitive hashing (LSH) [12, 13] are common. JUNO++ is compatible with various indexing methods, including Flat, IVF, and HNSW. 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:21 

_Encoding._ Encoding methods aim to minimize the memory usage of search points. A widely used approach is product quantization (PQ)[29], which divides the space into subspaces and encodes the projections of search points. Several techniques, such as DPQ[33] and OPQ[20], optimize the codebook to improve search quality. Scalar quantization (SQ)[72] encodes vector components independently and linearly, similar to traditional quantization in deep neural networks (DNNs)[22, 23]. Additive quantization (AQ)[4] represents search points as the sum of codebook entries. Currently, JUNO++ supports only product quantization (PQ). 

_Hardware Acceleration._ Specialized architectures include hardware support for hierarchical product quantization [1] and high-performance k-selection [70]. ANNA proposed an end-to-end hardware solution for PQ-based ANN search [35]. Tree-based designs for low-dimensional ANN search have also been explored [9, 69]. In addition to computation, large-scale ANN search challenges memory and storage subsystems. DiskANN uses graph-based indexing for limited RAM and SSDs [28], while SPANN combines memory and disk indexing [10]. These methods partition large datasets to fit within resource limits, particularly when GPU memory cannot handle the entire dataset. Thus, we evaluate JUNO++ with a 100M dataset to demonstrate its superior performance. Several approaches have mapped ANN search to existing hardware, such as ScANN using AVX on CPUs [24] and RTNN leveraging the RT core for low-dimensional searches [40, 73]. Our system aims to utilize the RT core for more general high-dimensional ANN search. 

## **8 Conclusion** 

This article introduces JUNO++, an end-to-end ANN search system that integrates a sparsity-aware codebook selection algorithm and an optimized RT core mapping. The core of our approach lies in leveraging sparsity and spatial locality, identified through comprehensive profiling. We utilize a distance threshold filtering technique that efficiently maps to RT cores. Moreover, the system is enhanced with time-based hit distance computation, aggressive hit count approximation, and Tensor-RT core pipelining. Evaluation of JUNO++ across multiple datasets shows a 2.1× to 8.5× improvement in search throughput compared to existing PQ-based ANN search methods. We also integrate our RT based acceleration into LLM to accelerate retrieval based sparse attention, by replacing the original 𝑞× 𝑘.𝑇 process with our RT version. Evaluation shows a 46% latency decrease with identical accuracy. Our solution has the potential to cooperate with already proposed sparse softmax mechanism to deliver higher end-to-end acceleration. 

## **Acknowledgments** 

We would like to thank the anonymous reviewers for their constructive feedback and comments to improve this work. Any opinions, findings, and conclusions in this article are those of the authors only and do not necessarily reflect the views of our sponsors. 

## **References** 

> [1] Ameer MS Abdelhadi, Christos-Savvas Bouganis, and George A Constantinides. 2019. Accelerated approximate nearest neighbors search through hierarchical product quantization. In _Proceedings of the 2019 International Conference on Field-Programmable Technology (ICFPT)_ . IEEE, 90–98. 

> [2] Jason Ansel, Edward Z. Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, C. K. Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Shunting Zhang, Michael Suo, Phil Tillet, Xu Zhao, Eikan Wang, Keren Zhou, Richard Zou, Xiaodong Wang, Ajit Mathews, William Wen, Gregory Chanan, Peng Wu, and Soumith Chintala. 2024. PyTorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In _Proceedings of the 29th ACM International Conference on Architectural Support_ 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:22 

_for Programming Languages and Operating Systems, Volume 2, ASPLOS 2024, La Jolla, CA, USA, 27 April 2024- 1 May 2024_ . ACM, 929–947. DOI:10.1145/3620665.3640366 

- [3] Oron Ashual, Shelly Sheynin, Adam Polyak, Uriel Singer, Oran Gafni, Eliya Nachmani, and Yaniv Taigman. 2022. KNN-diffusion: Image generation via large-scale retrieval. _CoRR_ abs/2204.02849 (2022). DOI:10.48550/arXiv.2204.02849 

- [4] Artem Babenko and Victor S. Lempitsky. 2014. Additive quantization for extreme vector compression. In _Proceedings of the 2014 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2014, Columbus, OH, USA, June 23-28, 2014_ . IEEE Computer Society. DOI:10.1109/CVPR.2014.124 

- [5] Artem Babenko and Victor S. Lempitsky. 2016. Efficient indexing of billion-scale datasets of deep descriptors. In _Proceedings of the 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016_ . IEEE Computer Society. DOI:10.1109/CVPR.2016.226 

- [6] Dmitry Baranchuk, Dmitry Persiyanov, Anton Sinitsin, and Artem Babenko. 2019. Learning to route in similarity graphs. In _Proceedings of the International Conference on Machine Learning_ . PMLR. 

- [7] Jeffrey S. Beis and David G. Lowe. 1997. Shape indexing using approximate nearest-neighbour search in highdimensional spaces. In _Proceedings of the 1997 Conference on Computer Vision and Pattern Recognition (CVPR’97), June 17-19, 1997, San Juan, Puerto Rico_ . IEEE Computer Society. DOI:10.1109/CVPR.1997.609451 

- [8] Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew R. Gormley. 2023. Unlimiformer: Long-range transformers with unlimited length input. _CoRR_ abs/2305.01625 (2023). DOI:10.48550/arXiv.2305.01625 

- [9] Faquan Chen, Rendong Ying, Jianwei Xue, Fei Wen, and Peilin Liu. 2023. ParallelNN: A parallel octree-based nearest neighbor search accelerator for 3D point clouds. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture, HPCA 2023, Montreal, QC, Canada, February 25 - March 1, 2023_ . IEEE. DOI:10.1109/HPCA56546.2023.10070940 

- [10] Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong Li, Mao Yang, and Jingdong Wang. 2021. Spann: Highly-efficient billion-scale approximate nearest neighborhood search. _Advances in Neural Information Processing Systems_ 34 (2021), 5199–5212. 

- [11] Rihan Chen, Bin Liu, Han Zhu, Yaoxuan Wang, Qi Li, Buting Ma, Qingbo Hua, Jun Jiang, Yunlong Xu, Hongbo Deng, and Bo Zheng. 2022. Approximate nearest neighbor search under neural similarity metric for large-scale recommendation. In _Proceedings of the 31st ACM International Conference on Information & Knowledge Management, Atlanta, GA, USA, October 17-21, 2022_ , Mohammad Al Hasan and Li Xiong (Eds.). ACM. DOI:10.1145/3511808.3557098 

- [12] Anirban Dasgupta, Ravi Kumar, and Tamás Sarlós. 2011. Fast locality-sensitive hashing. In _Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ . 

- [13] Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab S Mirrokni. 2004. Locality-sensitive hashing scheme based on p-stable distributions. In _Proceedings of the 20th Annual Symposium on Computational Geometry_ . 

- [14] Min Dong, Zhe Wang, Chenghui Dong, Xiaomin Mu, and Yide Ma. 2017. Classification of region of interest in mammograms using dual contourlet transform and improved KNN. _Journal of Sensors_ 2017 (2017). DOI:10.1155/2017/3213680 

- [15] Wei Dong, Charikar Moses, and Kai Li. 2011. Efficient k-nearest neighbor graph construction for generic similarity measures. In _Proceedings of the 20th International Conference on World Wide Web_ . 

- [16] J. Elseberg, S. Magnenat, R. Siegwart, and A. Nüchter. 2012. Comparison of nearest-neighbor-search strategies and implementations for efficient shape registration. _Journal of Software Engineering for Robotics (JOSER)_ 3, 1 (2012). 

- [17] Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. 2017. Fast approximate nearest neighbor search with the navigating spreading-out graph. _arXiv preprint arXiv:1707.00143_ (2017). 

- [18] Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. 2019. Fast approximate nearest neighbor search with the navigating spreading-out graph. _Proceedings of the VLDB Endow._ 12, 5 (jan 2019), 14 pages. DOI:10.14778/3303753.3303754 

- [19] Jianyang Gao and Cheng Long. 2023. High-dimensional approximate nearest neighbor search: with reliable and efficient distance comparison operations. _CoRR_ abs/2303.09855 (2023). DOI:10.48550/arXiv.2303.09855 

- [20] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product quantization. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 36, 4 (2013). 

- [21] Fabian Groh, Lukas Ruppert, Patrick Wieschollek, and Hendrik P. A. Lensch. 2023. GGNN: Graph-based GPU nearest neighbor search. _IEEE Transactions Big Data_ 9, 1 (2023). DOI:10.1109/TBDATA.2022.3161156 

- [22] Cong Guo, Jiaming Tang, Weiming Hu, Jingwen Leng, Chen Zhang, Fan Yang, Yunxin Liu, Minyi Guo, and Yuhao Zhu. 2023. OliVe: Accelerating large language models via hardware-friendly outlier-victim pair quantization. In _Proceedings of the 50th Annual International Symposium on Computer Architecture, ISCA 2023, Orlando, FL, USA, June 17-21, 2023_ , Yan Solihin and Mark A. Heinrich (Eds.). ACM. DOI:10.1145/3579371.3589038 

- [23] Cong Guo, Chen Zhang, Jingwen Leng, Zihan Liu, Fan Yang, Yunxin Liu, Minyi Guo, and Yuhao Zhu. 2022. ANT: Exploiting adaptive numerical data type for low-bit deep neural network quantization. In _Proceedings of the 55th IEEE/ACM International Symposium on Microarchitecture, MICRO 2022, Chicago, IL, USA, October 1-5, 2022_ . IEEE. DOI:10.1109/MICRO56248.2022.00095 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:23 

- [24] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. 2020. Accelerating large-scale inference with anisotropic vector quantization. In _Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event (Proceedings of Machine Learning Research, Vol. 119)_ . PMLR. Retrieved from http://proceedings.mlr.press/v119/guo20h.html 

- [25] Eric Haines and Tomas Akenine-Möller (Eds.). 2019. _Ray Tracing Gems_ . Apress. Retrieved from http://raytracinggems. com 

- [26] Kiana Hajebi, Yasin Abbasi-Yadkori, Hossein Shahbazi, and Hong Zhang. 2011. Fast approximate nearest-neighbor search with k-nearest neighbor graph. In _Proceedings of the 22nd International Joint Conference on Artificial Intelligence_ . 

- [27] Masajiro Iwasaki and Daisuke Miyazaki. 2018. Optimization of indexing based on k-nearest neighbor graph for proximity search in high-dimensional data. _arXiv preprint arXiv:1810.07355_ (2018). 

- [28] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar Krishnawamy, and Rohan Kadekodi. 2019. Diskann: Fast accurate billion-point nearest neighbor search on a single node. _Advances in Neural Information Processing Systems_ 32 (2019). 

- [29] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 33, 1 (2010). 

- [30] Hervé Jégou, Romain Tavenard, Matthijs Douze, and Laurent Amsaleg. 2011. Searching in one billion vectors: Re-rank with source coding. In _Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing, ICASSP 2011, May 22-27, 2011, Prague Congress Center, Prague, Czech Republic_ . IEEE. DOI:10.1109/ICASSP.2011.5946540 

- [31] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. _IEEE Transactions on Big Data_ 7, 3 (2019), 535–547. 

- [32] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. 2020. Reformer: The efficient transformer. In _Proceedings of the 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net. 

- [33] Benjamin Klein and Lior Wolf. 2019. End-to-end supervised product quantization for image search and retrieval. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 

- [34] Jon M. Kleinberg. 1997. Two algorithms for nearest-neighbor search in high dimensions. In _Proceedings of the 29th Annual ACM Symposium on the Theory of Computing, El Paso, Texas, USA, May 4-6, 1997_ , Frank Thomson Leighton and Peter W. Shor (Eds.). ACM. DOI:10.1145/258533.258653 

- [35] Yejin Lee, Hyunji Choi, Sunhong Min, Hyunseung Lee, Sangwon Beak, Dawoon Jeong, Jae W. Lee, and Tae Jun Ham. 2022. ANNA: Specialized architecture for approximate nearest neighbor search. In _Proceedings of the IEEE International Symposium on High-Performance Computer Architecture, HPCA 2022, Seoul, South Korea, April 2-6, 2022_ . IEEE. DOI:10.1109/HPCA53966.2022.00021 

- [36] Di Liu, Meng Chen, Baotong Lu, Huiqiang Jiang, Zhenhua Han, Qianxi Zhang, Qi Chen, Chengruidong Zhang, Bailu Ding, Kai Zhang, Chen Chen, Fan Yang, Yuqing Yang, and Lili Qiu. 2024. RetrievalAttention: Accelerating long-context LLM inference via vector retrieval. _CoRR_ abs/2409.10516 (2024). DOI:10.48550/ARXIV.2409.10516 

- [37] Yuchen Liu, Zhibin Pan, Liangzhuang Wang, and Yang Wang. 2022. A new fast inverted file-based algorithm for approximate nearest neighbor search without accuracy reduction. _Information Sciences_ 608 (2022). DOI:10.1016/j.ins.2022.06.086 

- [38] Zihan Liu, Jingwen Leng, Zhihui Zhang, Quan Chen, Chao Li, and Minyi Guo. 2022. VELTAIR: towards highperformance multi-tenant deep learning services via adaptive compilation and scheduling. In _ASPLOS ’22: 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Lausanne, Switzerland, 28 February 2022 - 4 March 2022_ , Babak Falsafi, Michael Ferdman, Shan Lu, and Thomas F. Wenisch (Eds.). ACM, 388–401. DOI:10.1145/3503222.3507752 

- [39] Zihan Liu, Xinhao Luo, Junxian Guo, Wentao Ni, Yangjie Zhou, Yue Guan, Cong Guo, Weihao Cui, Yu Feng, Minyi Guo, Yuhao Zhu, Minjia Zhang, Chen Jin, and Jingwen Leng. 2025. VQ-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference. In _IEEE International Symposium on High Performance Computer Architecture, HPCA 2025, Las Vegas, NV, USA, March 1-5, 2025_ . IEEE, 1496–1509. DOI:10.1109/HPCA61900.2025.00112 

- [40] Zihan Liu, Wentao Ni, Jingwen Leng, Yu Feng, Cong Guo, Quan Chen, Chao Li, Minyi Guo, and Yuhao Zhu. 2024. JUNO: Optimizing High-Dimensional Approximate Nearest Neighbour Search with Sparsity-Aware Algorithm and RayTracing Core Mapping. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS 2024, La Jolla, CA, USA, 27 April 2024- 1 May 2024_ , Rajiv Gupta, Nael B. Abu-Ghazaleh, Madan Musuvathi, and Dan Tsafrir (Eds.). ACM, 549–565. DOI:10.1145/3620665.3640360 

- [41] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 42, 4 (2018). 

- [42] Adam Marrs, Peter Shirley, and Ingo Wald (Eds.). 2021. _Ray Tracing Gems II_ . Apress. Retrieved from http:// raytracinggems.com/rtg2 

- [43] Meta. 2023. Indexing 1G vectors. Retrieved from https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

Z. Liu et al. 

133:24 

- [44] Marius Muja and David G Lowe. 2014. Scalable nearest neighbor algorithms for high dimensional data. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 36, 11 (2014). 

- [45] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii, Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko, Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong, and Misha Smelyanskiy. 2019. Deep learning recommendation model for personalization and recommendation systems. _CoRR_ abs/1906.00091 (2019). arXiv:1906.00091 http://arxiv. org/abs/1906.00091 

- [46] NVIDIA. [n. d.]. NVIDIA OptiX™Ray Tracing Engine. Retrieved from https://developer.nvidia.com/rtx/ray-tracing/ optix 

- [47] NVIDIA. 2018. NVIDIA TURING GPU ARCHITECTURE. Retrieved from https://images.nvidia.com/aem-dam/enzz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf 

- [48] NVIDIA. 2021. NVIDIA AMPERE GA102 GPU ARCHITECTURE. Retrieved from https://www.nvidia.com/content/ PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf 

- [49] NVIDIA. 2022. NVIDIA ADA CRAFT The engineering marvel of the RTX 4090. Retrieved from https://images.nvidia. com/aem-dam/Solutions/geforce/ada/ada-lovelace-architecture/nvidia-ada-gpu-craft.pdf 

- [50] NVIDIA. 2023. NVIDIA ADA GPU ARCHITECTURE. Retrieved from https://images.nvidia.com/aem-dam/Solutions/ Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.0.pdf 

- [51] NVIDIA. 2023. NVIDIA Tensor Cores Unprecedented Acceleration for HPC and AI. Retrieved from https://www.nvidia. com/en-us/data-center/tensor-cores/ 

- [52] OpenAI. 2023. GPT-4 technical report. _CoRR_ abs/2303.08774 (2023). DOI:10.48550/arXiv.2303.08774 

- [53] Zhen Peng, Minjia Zhang, Kai Li, Ruoming Jin, and Bin Ren. 2023. iQAN: Fast and accurate vector search with efficient intra-query parallelism on multi-core architectures. In _Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming, PPoPP 2023, Montreal, QC, Canada, 25 February 2023 - 1 March 2023_ , Maryam Mehri Dehnavi, Milind Kulkarni, and Sriram Krishnamoorthy (Eds.). ACM. DOI:10.1145/3572848.3577527 

- [54] Sakti Pramanik and Jinhua Li. 1999. Fast approximate search algorithm for nearest neighbor queries in high dimensions. In _Proceedings of the 15th International Conference on Data Engineering, Sydney, Australia, March 23-26, 1999_ , Masaru Kitsuregawa, Michael P. Papazoglou, and Calton Pu (Eds.). IEEE Computer Society. DOI:10.1109/ICDE.1999.754931 

- [55] Charles Ruizhongtai Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. 2017. PointNet: Deep learning on point sets for 3D classification and segmentation. In _Proceedings of the 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017_ . IEEE Computer Society. DOI:10.1109/CVPR.2017.16 

- [56] Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, and Xinran Xu. 2024. Mooncake: A KVCache-centric disaggregated architecture for LLM serving. _CoRR_ abs/2407.00079 (2024). DOI:0.48550/ ARXIV.2407.00079 

- [57] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of Machine Learning Research_ 21, 140 (2020), 1–67. 

- [58] Yandex Research. 2021. Benchmarks for Billion-Scale Similarity Search. Retrieved from https://research.yandex.com/ blog/benchmarks-for-billion-scale-similarity-search 

- [59] Michael Shen, Muhammad Umar, Kiwan Maeng, G. Edward Suh, and Udit Gupta. 2025. Hermes: Algorithm-system codesign for efficient retrieval-augmented generation at-scale. In _Proceedings of the 52nd Annual International Symposium on Computer Architecture, ISCA 2025, Tokyo, Japan, June 21-25, 2025_ . ACM, 958–973. DOI:10.1145/3695053.3731076 

- [60] Harsha Vardhan Simhadri, Ravishankar Krishnaswamy, Gopal Srinivasa, Suhas Jayaram Subramanya, Andrija Antonijevic, Dax Pryce, David Kaczynski, Shane Williams, Siddarth Gollapudi, Varun Sivashankar, Neel Karia, Aditi Singh, Shikhar Jaiswal, Neelam Mahapatro, Philip Adams, Bryan Tower, and Yash Patel. [n. d.]. 

- [61] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and efficient foundation language models. _CoRR_ abs/2302.13971 (2023). DOI:10.48550/arXiv.2302.13971 

- [62] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Proceedings of the Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (Eds.). 

- [63] Ingo Wald and Vlastimil Havran. 2006. On building fast kd-trees for ray tracing, and on doing that in O(N log N). In _Proceedings of the 2006 IEEE Symposium on Interactive Ray Tracing_ . 61–69. DOI:10.1109/RT.2006.280216 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

## JUNO++: Optimizing ANNS and Enabling Efficient Sparse Attention in LLM via RT Core 133:25 

- [64] Jing Wang, Jingdong Wang, Gang Zeng, Zhuowen Tu, Rui Gan, and Shipeng Li. 2012. Scalable k-nn graph construction for visual descriptors. In _Proceedings of the 2012 IEEE Conference on Computer Vision and Pattern Recognition_ . IEEE. 

- [65] Liwei Wang, Yan Zhang, and Jufu Feng. 2005. On the euclidean distance of images. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 27, 8 (2005). DOI:10.1109/TPAMI.2005.165 

- [66] Mengzhao Wang, Xiaoliang Xu, Qiang Yue, and Yuxiang Wang. 2021. A comprehensive survey and experimental comparison of graph-based approximate nearest neighbor search. _Proceedings of the VLDB Endow._ 14, 11 (2021). DOI:10.14778/3476249.3476255 

- [67] Ruoxi Wang, Rakesh Shivanna, Derek Zhiyuan Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed H. Chi. 2021. DCN V2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In _Proceedings of the WWW’21: The Web Conference 2021, Virtual Event / Ljubljana, Slovenia, April 19-23, 2021_ , Jure Leskovec, Marko Grobelnik, Marc Najork, Jie Tang, and Leila Zia (Eds.). ACM / IW3C2. DOI:10.1145/3442381.3450078 

- [68] Wenxuan Wu, Zhongang Qi, and Fuxin Li. 2019. PointConv: Deep convolutional networks on 3D point clouds. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019_ . Computer Vision Foundation / IEEE. DOI:10.1109/CVPR.2019.00985 

- [69] Tiancheng Xu, Boyuan Tian, and Yuhao Zhu. 2019. Tigris: Architecture and algorithms for 3D perception in point clouds. In _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, MICRO 2019, Columbus, OH, USA, October 12-16, 2019_ . ACM. DOI:10.1145/3352460.3358259 

- [70] Jialiang Zhang, Soroosh Khoram, and Jing Li. 2018. Efficient large-scale approximate nearest neighbor search on OpenCL FPGA. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ . 4924–4932. 

- [71] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W. Barrett, and Ying Sheng. 2024. SGLang: Efficient execution of structured language model programs. In _Proceedings of the Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024_ . 

- [72] Wengang Zhou, Yijuan Lu, Houqiang Li, and Qi Tian. 2012. Scalar quantization for large scale image search. In _Proceedings of the 20th ACM Multimedia Conference, MM’12, Nara, Japan, October 29 - November 02, 2012_ , Noboru Babaguchi, Kiyoharu Aizawa, John R. Smith, Shin’ichi Satoh, Thomas Plagemann, Xian-Sheng Hua, and Rong Yan (Eds.). ACM. DOI:10.1145/2393347.2393377 

- [73] Yuhao Zhu. 2022. RTNN: Accelerating neighbor search using hardware ray tracing. In _Proceedings of the PPoPP’22: 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, Seoul, Republic of Korea, April 2 - 6, 2022_ , Jaejin Lee, Kunal Agrawal, and Michael F. Spear (Eds.). ACM. 

Received 28 May 2025; revised 10 August 2025; accepted 10 September 2025 

ACM Trans. Arch. Code Optim., Vol. 22, No. 4, Article 133. Publication date: December 2025. 

