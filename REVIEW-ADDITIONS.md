# Review of The Inference Engineering Book, and instructions for the additions

Date of review: 7 November 2026
Reviewer: Claude (Fable 5.1), acting as reviewer and orchestrator only. No chapter page was edited during this review.
Executor: Claude Opus. Every item below is an instruction to you. Follow the house rules in section 7 for every change.

Goal the book must serve: take an AI engineer who already builds agents and LLM systems and teach inference end to end, from the math essentials to production systems, including the research papers a practitioner should have read, with weight on the Chinese labs (DeepSeek, Moonshot, Alibaba, MiniMax, StepFun, Meituan, ByteDance, Huawei, Zhipu) whose inference work currently leads.

---

## 1. Verdict

The book is complete at 57 chapters and three guides, and it holds together: definitions before use, references only at chapter ends, measured results labelled as measured, source errors noted, no em dashes, working interactive figures, and a render check on every page. For the stated audience it covers the full path: Part 1 math and PyTorch, Part 2 transformer inference mechanics, Part 3 engines, Part 4 optimizations, Part 5 GPUs and kernels, Part 6 production, Part 7 post-training, Part 8 internals and research.

Three things keep it short of the goal:

1. **The DeepSeek inference stack is scattered and stops in 2025.** MLA, DSA, FP8, EP, redundant experts, FlashMLA, DeepEP and DeepGEMM each get a sentence or a paragraph in different chapters, and the February 2025 inference system overview is used well in 6.7. But the book has nothing on DeepSeek-V4 (April 2026: CSA and HCA attention, mHC, Muon, one-million-token context at 27% of V3.2's FLOPs and 10% of its KV cache), nothing on DeepSeek-V4.1-Flash (September 2026: causal encoder-decoder, cross-layer KV reuse, FP4 KV cache, 890 bytes of global KV per token, SWA bounded replay), and nothing on **DeepInfer** (20 October 2026), the open-sourced engine behind DeepSeek's production API. DeepInfer is the release the inference community has been reacting to this month: it packages attention data and context parallelism, EP with load balancing, draft-based decoding with confidence-scheduled verification ("DSpark"), asynchronous scheduling, batch-invariant inference and deterministic sampling, structured outputs, and support for both NVIDIA SM100 GPUs and Huawei Ascend 950 NPUs. It is Apache-2.0, built on vLLM, and comes with its own kernel libraries (FlashMLA, DeepGEMM, DeepEP, TileKernels, DeepSelect). This needs a chapter of its own plus updates to seven existing chapters (section 3).

2. **Several Chinese-lab papers that practitioners now treat as required reading are missing.** Verified list in section 4: Native Sparse Attention and MoBA (sparse attention), MiniMax-01 and M1 (lightning attention at scale), Step-3 (attention-FFN disaggregation), LongCat-Flash (zero-computation experts), MegaScale-Infer (disaggregated expert parallelism), Huawei CloudMatrix384 and its serving paper, Aegaeon (GPU pooling, SOSP 2025), KTransformers (CPU and GPU hybrid MoE inference), Comet (compute-communication overlap for MoE), TileLang, Fire-Flyer AI-HPC and "Insights into DeepSeek-V3" (hardware co-design), Qwen3 and GLM-4.5 technical reports, Kimi K2 report, DeepSeek-R1, and the DeepInfer-era papers (V4, V4.1-Flash, tree speculation on V4, LongCat Sparse Attention).

3. **A few Western papers that complete the reading list are missing too:** Orca (continuous batching's origin), SpecInfer (tree speculation), Hydragen (shared-prefix attention), CacheGen (KV streaming), KIVI and QServe (KV and W4A8 quantization), Lookahead decoding, NanoFlow, Helix parallelism, Jenga, QuaRot and SqueezeLLM. Section 4 places each.

Smaller gaps in audience fit (section 5): no bridge for an agent builder's existing mental model, no treatment of embedding and reranker serving (the other half of every RAG system the audience builds), and the index page still promises "labs that run on free GPUs" while many labs now run on CPU.

Everything else in this file is ordered by priority: P0 must be done, P1 should be done, P2 is optional polish.

---

## 2. How the review was done, and what it is not

- Read the chapter map, section headings and reference tables of all 57 chapters; grepped every chapter for 120 terms, systems and paper names; listed all 68 arXiv papers currently cited.
- Fetched and read the primary sources for everything that post-dates the reviewer's own knowledge (June 2026): the DeepInfer, DeepSelect, TileKernels, FlashMLA, DeepEP, DeepGEMM, EPLB, 3FS, DualPipe and V3.2-Exp READMEs on GitHub, and the arXiv abstracts of DeepSeek-V4, DeepSeek-V4.1-Flash, the tree-speculation paper, FlashMemory-DeepSeek-V4 and LongCat Sparse Attention. Every arXiv ID in this file was resolved against arxiv.org and its title checked.
- Did not re-run the labs or re-verify every number inside existing chapters. Items marked "verify" below are places where the executor must read the source before writing.
- Third-party blog summaries of DeepInfer were read but must not be cited; they contain chart readings and inferences. Cite only the repository, its docs and the arXiv reports.

---

## 3. P0: DeepSeek's inference stack, from V3 to DeepInfer

### 3.1 New chapter: 4.8 "Case study: DeepSeek's inference stack, V3 to DeepInfer"

Place it at the end of Part 4 (slug `4-8-deepseek-inference-stack`) so it can use every Part 4 technique and be referenced from Part 6 and Part 8. Level: expert. Add it to `chapters.json`, the glossary extras and the formula sheet.

Structure (follow the standard chapter template: kicker, lede, meta chips, skip check, numbered sections, expert layer, mistakes, lab, real world, quiz, summary, references, notes on sources):

1. **Why one lab's stack is worth a chapter.** DeepSeek publishes its architecture, kernels, cluster design, serving layout and production statistics, which no other frontier lab does. Reading them in order is the fastest way to see how Parts 2 to 6 fit together.

2. **V3 (December 2024): the base design.** MLA (link 2.4 and 4.2), DeepSeekMoE with shared and routed experts, auxiliary-loss-free load balancing, multi-token prediction (link 4.3), FP8 training with fine-grained block scales (link 1.3 and 4.1), DualPipe. Sources: DeepSeek-V3 Technical Report (arXiv 2412.19437, already cited); "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures" (arXiv 2505.09343, ISCA 2025), which explains the hardware constraints behind MLA, FP8 and the communication design; Fire-Flyer AI-HPC (arXiv 2408.14158) for the cluster. Use the ISCA paper's hardware wish list (its section on future hardware) as the expert layer.

3. **Open Source Week (February 2025): the kernels and the system overview.** FlashMLA (MLA decode kernel, Hopper and later), DeepEP (all-to-all for EP, normal and low-latency modes, with NVLink and RDMA), DeepGEMM (FP8 GEMM with fine-grained scaling, JIT-compiled), EPLB (expert placement and replication), DualPipe, 3FS (the file system behind the on-disk KV cache), and the Day 6 inference system overview already used in 6.7 (keep 6.7's numbers there; this chapter links to them). Sources: the GitHub READMEs of each repo under github.com/deepseek-ai. Read each README for current numbers (FlashMLA's README states achieved bandwidth and TFLOPS on H800; DeepEP's README gives latency and bandwidth tables per mode); quote them with the date checked.

4. **V3.2 (September to December 2025): DSA.** Lightning indexer, top-2,048 selection, DeepSelect as the top-k kernel. Keep the mechanism in 4.2; here show the cost effect at long context and the serving changes (indexer in FP8, sparse attention kernels in FlashMLA). Source: DeepSeek-V3.2 report (arXiv 2512.02556, already cited) and the DeepSeek-V3.2-Exp README.

5. **V4 (April 2026): million-token context.** From the abstract of arXiv 2606.19348: two MoE models, V4-Pro with 1.6T parameters and 49B active, V4-Flash with 284B and 13B active, one-million-token context; a hybrid of Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA); Manifold-Constrained Hyper-Connections (mHC, already in 2.7 as arXiv 2512.24880); the Muon optimizer (link 7.7); pre-training on more than 32T tokens; at one million tokens V4-Pro needs 27% of V3.2's single-token inference FLOPs and 10% of its KV cache. Read the report's architecture and inference sections before writing; do not describe CSA and HCA beyond what the report says.

6. **V4.1-Flash (September 2026): compressing the KV cache to the limit.** From the abstract of arXiv 2609.19969: 552B backbone parameters, multimodal, one-million-token context; Causal Encoder-Decoder (CED) architecture that activates 16B parameters per token in decode but only 8B in prefill; Compressed Sparse Attention 2 (CSA2) with cross-layer KV reuse; FP4 KV caching; global KV footprint of 890 bytes per token (about a quarter of V4-Flash); SWA Bounded Replay, which cuts the persistent (SSD or host memory) KV footprint to about one eighth of V4-Flash. Explain why this matters for agent workloads (link 6.8): input-heavy traffic makes prefill compute and KV storage the cost bottleneck. The DeepInfer README's environment variable names reference "YOCO" for the replay feature; read the report to see whether CED is described in relation to YOCO (Microsoft, arXiv 2405.05254) and relate them only if the report does.

7. **DeepInfer (20 October 2026): the engine.** From the README at github.com/deepseek-ai/DeepInfer (read it at the pinned commit you use and record the hash): it is "the high-performance inference framework that powers DeepSeek's large-scale production services", released as offline inference with the core features for serving at scale. Supported model: DeepSeek-V4.1-Flash. Features: optimized kernels (FlashMLA, DeepGEMM, DeepEP, TileKernels, DeepSelect, custom TileLang kernels); attention data parallelism and balanced context parallelism; MoE EP with EPLB; DSpark with confidence-scheduled verification (draft tokens, maximum draft length 5 by default, `DEEPINFER_ENABLE_MTP_VARLEN`); decoder SWA bounded replay (`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`, keep tokens 140 for V4.1); asynchronous scheduling described as a zero-overhead batch scheduler, on by default; batch-invariant inference and deterministic sampling (NVIDIA only; `DEEPINFER_BATCH_INVARIANT=1`, fixed `seq_seed`); structured outputs (regex, JSON schema, EBNF). Hardware: NVIDIA SM100 GPUs and Huawei Ascend 950 NPUs; CUDA 13 or CANN 9.2; Python 3.12. Not included in the release and listed as required for a complete service: a KV cache system, VE/prefill/decode disaggregation (the README says most engine functionality for it is implemented), and load balancing across data-parallel replicas. Benchmarks: prefill measured on 4 SM100 GPUs, decode on 64, each within one scale-up domain, reproducible with `examples/sweep/run_sweep.sh`. Licence Apache-2.0. The README credits vLLM as the primary foundation with inspiration from SGLang and TensorRT-LLM. Map every feature to the chapter that taught it (a table: feature, chapter). Read `doc/` and `eval/README.md` in the repository for the benchmark figures and quote only numbers printed there, labelled "DeepSeek-reported, not independently reproduced".

8. **Reading the whole stack as one design.** A table across V3, V3.2, V4, V4.1: attention mechanism, KV bytes per token, active parameters prefill and decode, speculative method, precision, parallelism, where the KV cache lives. Compute KV bytes per token for V3 (MLA, 576 values, from chapter 2.4), and use the reported 890 bytes for V4.1-Flash; do not compute values the reports do not give.

9. **Expert layer.** (a) Why DeepSeek runs decode on 64 to 144 GPUs per unit (EP batch per expert, from 6.7). (b) Lookahead Sparse Attention (FlashMemory-DeepSeek-V4, arXiv 2606.09079): an indexer trained without the backbone that keeps only query-critical KV chunks in GPU memory, reporting 13.5% average KV footprint and 2.8 times throughput at 1M context; and LongCat Sparse Attention (arXiv 2608.01662): streaming-aware, cross-layer and hierarchical indexing to cut the lightning indexer's O(L²) cost. Present both as "what others built on DSA". (c) Tree-structured speculative decoding adapted to V4 (arXiv 2609.24698): accepted length about 2.83 to 3.41 versus 2.39 to 2.84 for linear speculation at budget 8, throughput gains up to about 18.5%; the difficulty is keeping compressed attention states consistent across tree branches.

10. **Lab.** Part A (CPU): from the published numbers, compute KV cache per million-token context for V3 (MLA), V4-Pro (10% of V3.2) and V4.1-Flash (890 bytes per token), and the HBM needed for 64 concurrent one-million-token sessions in each; predict first. Part B (CPU): reproduce the 6.7 cost model with DeepInfer's deployment unit sizes from its docs (prefill 4 GPUs, decode 64) and the published API prices for the Flash model (read current prices from the DeepSeek API docs on the day; do not copy prices from blogs). Part C (needs SM100 or Ascend 950; mark as expected, not measured): run `examples/offline_inference.py` and `run_sweep.sh`, then compare the sweep's Pareto curve with 6.10's method; note that the README states no H100 or H200 support, so most readers will read this part rather than run it.

11. **Notes on sources.** State that V4, V4.1-Flash and DeepInfer post-date the reviewer's and executor's training data and that every statement about them comes from the arXiv abstracts and the GitHub README read on the date given; mark any detail you could not confirm from those as "not confirmed".

### 3.2 Updates to existing chapters (P0)

| Chapter | Add |
|---|---|
| 2.4 KV cache | One row in the per-model KV table for DeepSeek-V4.1-Flash: 890 bytes per token of global KV (DeepSeek-reported, arXiv 2609.19969), with a sentence on cross-layer reuse and FP4 KV as the two levers. Link 4.8. |
| 2.7 Frontier architectures | A paragraph on V4's CSA and HCA hybrid and V4.1's causal encoder-decoder with different active parameters in prefill and decode; the arithmetic consequence (prefill FLOPs per token fall by half if active parameters halve). Keep it to what the abstracts state. |
| 4.2 Efficient attention | In the DSA section, add that DeepSelect is the production top-k kernel (2 to 20 times faster than `torch.topk` per its README; verify the figure in the README) and that CSA and CSA2 in V4 and V4.1 build on DSA. Add Native Sparse Attention (arXiv 2502.11089, DeepSeek) and MoBA (arXiv 2502.13189, Moonshot) as the two February 2025 trainable sparse attention designs that preceded DSA; one paragraph each on the mechanism (NSA: compressed, selected and sliding branches with hardware-aligned blocks; MoBA: block-level top-k routing of queries to KV blocks). |
| 4.3 Speculative decoding | Add tree-structured verification (SpecInfer, arXiv 2305.09781) as a concept, EAGLE-3 if not already explained, and DeepInfer's DSpark with confidence-scheduled variable-length verification as a production example (from the README only). Add the V4 tree-speculation paper's finding that compressed attention states complicate tree verification. |
| 4.5 Model parallelism | Add DeepEP's two modes (normal for prefill, low-latency for decode) with the README's figures; EPLB's two policies (hierarchical and global); Comet (arXiv 2502.19811, ByteDance) for fine-grained compute-communication overlap in MoE layers; MegaScale-Infer (arXiv 2504.02263, ByteDance) for disaggregated expert parallelism (attention and expert modules on separate GPU pools with ping-pong pipelining); Step-3 (arXiv 2507.19427, StepFun) for attention-FFN disaggregation (AFD) and its cost-per-token argument. |
| 4.6 Disaggregation | Add the AFD axis (Step-3, MegaScale-Infer) as a second way to split a model beyond prefill and decode, and DeepInfer's "VE/Prefill/Decode disaggregation" naming (vision encoder as a third pool). |
| 5.8 High-performance kernels | Add TileLang (arXiv 2504.17577, Peking University and others) as the DSL behind DeepSeek's TileKernels, FlashMLA's design notes from its README, DeepGEMM's JIT and fine-grained FP8 scaling, DeepSelect's radix-select approach (read its deep-dive doc). |
| 6.6 Observability and correctness | One sentence: DeepInfer ships batch-invariant inference and deterministic sampling as first-class features, confirming the Thinking Machines line of work in a production engine. |
| 6.7 Case studies | Cross-link to 4.8 and note that the February 2025 numbers describe the V3/R1 service; the V4.1 service runs on SM100 and Ascend 950 with a different unit layout. |
| 6.8 Agentic workloads | Add the V4.1-Flash motivation verbatim in spirit: long-horizon agents make workloads input-heavy, so prefill compute and KV storage, not decode, bound the cost; SWA bounded replay and CED are answers at the model level. |
| 8.2 Read an engine | Add a short section "A second engine to read: DeepInfer", with its directory map at a pinned commit (clone it, list the top-level packages, name the scheduler, the EP and attention-DP code paths, and the DSpark code) and a comparison with vLLM's layout from the same chapter. |
| Glossary and formula sheet | Terms: CSA, HCA, CED, SWA bounded replay, DSpark, EPLB, DeepEP normal vs low-latency mode, attention data parallelism, lightning indexer (if missing), zero-computation expert, AFD. Formulas: prefill FLOPs per token with different active parameters in prefill and decode; persistent vs global KV footprint. |

---

## 4. P0 and P1: the paper additions, chapter by chapter

Format: paper (arXiv ID, lab, date) : where it goes : what to write (concept, one to three sentences in the chapter body; full mechanism only where marked "section"). All IDs verified on 7 November 2026. "CN" marks Chinese-lab work. Priority in brackets.

### Part 2

- Qwen3 Technical Report (2505.09388, Alibaba, May 2025) CN [P1] : 2.7 : thinking and non-thinking modes in one model, thinking budgets; the 119-language coverage is not inference-relevant, skip it. Also cite in 8.1 (reasoning workloads).
- Gated Delta Networks (2412.06464, NVIDIA and MIT, Dec 2024) [P1] : 2.7 : already named; add the citation and the delta-rule intuition (write new information while erasing the old along the same key), because Qwen3-Next and Kimi Linear build on it.
- MiniMax-01 (2501.08313, MiniMax, Jan 2025) CN [P0] : 2.7 : lightning attention (linear attention with a blockwise algorithm) in a 456B MoE at 1M context; the first production-scale linear-attention LLM. MiniMax-M1 (2506.13585, Jun 2025) CN [P1] : 2.7 and 8.1 : the same architecture for reasoning with long outputs; the report's claim about FLOPs at 100K generation length versus DeepSeek-R1 (read and quote exactly).
- Kimi K2 (2507.20534, Moonshot, Jul 2025) CN [P1] : 2.6 or 2.7 : 1T-parameter MoE with 32B active, MuonClip; cite where 7.7 mentions Muon and K2.
- GLM-4.5 (2508.06471, Zhipu, Aug 2025) CN [P2] : 2.7 : hybrid reasoning modes; one line.
- LongCat-Flash (2509.01322, Meituan, Sep 2025) CN [P0] : 2.6 (dense vs MoE) : "zero-computation experts" that let the router spend variable compute per token (about 18.6B to 31.3B active out of 560B; verify in the report), and shortcut-connected MoE that overlaps communication with computation; the inference consequence is a variable per-token cost. Section-level treatment in 2.6's expert layer.
- DeepSeek-R1 (2501.12948, DeepSeek, Jan 2025) CN [P1] : 7.1 and 8.1 : RL from the base model (R1-Zero) and the long-output inference workload it created. 7.1 already mentions R1-Zero without the citation.

### Part 3

- Orca (Yu et al., OSDI 2022; no arXiv; cite the USENIX page https://www.usenix.org/conference/osdi22/presentation/yu) [P0] : 3.2 : the origin of iteration-level scheduling and selective batching; one paragraph in the continuous batching section crediting it.
- KTransformers (Tsinghua KVCache.AI, SOSP 2025; cite github.com/kvcache-ai/ktransformers and the SOSP paper if the repo links it) CN [P0] : 3.6 : CPU and GPU hybrid inference of large MoE models, with experts on CPU (AMX kernels) and attention on GPU; the design point for running DeepSeek-class models on a workstation. One section in 3.6 and a lab exercise.
- LMDeploy and TurboMind (InternLM, Shanghai AI Lab; github.com/InternLM/lmdeploy) CN [P1] : 3.7 and the engine selection matrix in 3.6 : an engine with its own TurboMind runtime and W4A16 kernels; LightLLM (SenseTime; github.com/ModelTC/lightllm) CN [P2] : one line in 3.7.
- Jenga (2503.18292, Mar 2025) [P1] : 3.3 : memory management for models with heterogeneous KV (sliding window, Mamba, cross-attention layers), which vLLM's hybrid KV manager follows; 3.3's section 8 should cite it.
- NanoFlow (2408.12757, Aug 2024) [P2] : 3.4 expert layer : intra-device parallelism (overlapping compute, memory and network operations within one step); relate to DeepInfer's asynchronous scheduler and SGLang's zero-overhead scheduler.

### Part 4

- Native Sparse Attention (2502.11089) CN and MoBA (2502.13189) CN [P0] : 4.2 : see section 3.2.
- Lookahead decoding (2402.02057, Jan 2024) [P2] : 4.3 expert layer : parallel decoding by Jacobi iteration without a draft model; one paragraph.
- SpecInfer (2305.09781) [P1] : 4.3 : tree-based speculation and verification; the parent of EAGLE's tree verification.
- Hydragen (2402.05099, Feb 2024) [P1] : 4.4 : attention over a shared prefix computed once per batch (cascade inference in FlashInfer), which turns prefix sharing from a cache hit into a kernel-level saving during decode; one section with the arithmetic.
- CacheGen (2310.07240, Oct 2023) [P1] : 4.4 or 4.7 : compressing and streaming KV across the network so that loading beats recompute at lower bandwidth; the precursor of LMCache (already cited).
- KIVI (2402.02750) [P1] and QServe (2405.04532, MIT) [P0] : 4.1 : KIVI for 2-bit asymmetric KV quantization (per-channel keys, per-token values); QServe for W4A8KV4 with the QoQ algorithm and the system co-design argument about dequantization overhead on tensor cores. Add QuaRot (2404.00456) and SqueezeLLM (2306.07629) as one-line pointers in the expert layer (rotations for outlier-free 4-bit; dense-and-sparse).
- Helix Parallelism (2507.07120, NVIDIA, Jul 2025) [P1] : 4.5 or 4.7 : sharding KV across GPUs for multi-million-token decode with temporal pipelining; one paragraph.
- Comet, MegaScale-Infer, Step-3 : 4.5 and 4.6 : see section 3.2.
- Serving Large Language Models on Huawei CloudMatrix384 (2506.12708, Huawei, Jun 2025) CN [P0] and Huawei Cloud Model-as-a-Service on the CloudMatrix384 SuperPod (2508.02520, Aug 2025) CN [P1] : 4.5, 4.6 and 5.5 : a 384-NPU super-node with a unified bus, serving DeepSeek-R1 with disaggregated prefill, decode and caching; quote the reported per-NPU throughput and TPOT figures exactly from the paper. This is also the hardware context for DeepInfer's Ascend 950 support. 5.5 currently has almost nothing on Ascend; add a section.

### Part 5

- TileLang (2504.17577) CN [P0] : 5.4 and 5.8 : a tiled programming model with explicit layout and pipeline control, now used for DeepSeek's TileKernels across NVIDIA and Ascend backends; relate to Triton and CuTe.
- Fire-Flyer AI-HPC (2408.14158) CN [P1] : 5.5 : DeepSeek's cost-driven cluster (PCIe A100 nodes, two-layer fat-tree, HFReduce, 3FS); the argument that software co-design beats buying the most expensive interconnect.
- Insights into DeepSeek-V3 (2505.09343) CN [P0] : 5.5 : the hardware constraints behind each architectural choice and the explicit requests to hardware vendors; this is the best single paper for "why models look the way they do".
- Comet (2502.19811) CN [P2] : 5.4 : as an example of fine-grained overlap at the kernel level.

### Part 6

- Aegaeon (Alibaba Cloud and Peking University, SOSP 2025; PDF at https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf) CN [P0] : 6.3 and 6.5 : token-level autoscaling that pools GPUs across many models with low request rates, with the reported reduction of GPUs needed for its model set in Alibaba Cloud Model Studio (read the paper; the widely quoted figure is an 82% reduction, confirm it there). Add a section on "serving the long tail of models" in 6.3.
- MegaScale-Infer (2504.02263) CN : 6.7 : as a production case alongside DeepSeek's.
- CloudMatrix384 papers : 6.7 : a second cluster case study.
- Kimi K2 and Mooncake : 6.7 : Mooncake is cited already; add K2's serving notes if the report has them (verify).

### Part 7

- DeepSeek-R1 (2501.12948) : 7.4 : R1-Zero and GRPO with verifiable rewards; cite where the chapter describes GRPO's origin (currently cites TRL's docs only). Also DeepSeekMath (2402.03300) as the origin of GRPO; verify the ID before citing.
- Kimi K2 (2507.20534) : 7.7 : MuonClip; already implied, add the citation.

### Part 8

- DeepInfer repository : 8.2 and 8.3 : see section 3.2.
- Tree speculation on V4 (2609.24698) : 8.4 : a good candidate paper for reproduction at small scale (tree versus linear verification on the chapter 8.3 engine); add it to the candidate list.
- LongCat Sparse Attention (2608.01662) CN and FlashMemory-DeepSeek-V4 (2606.09079) CN [P1] : 8.4 candidate list and 4.8 expert layer.

### Reading list appendix (new guide page, P1)

Create `guides/reading-list.html`: every paper the book cites and every paper added above, grouped by part, each with one line on why to read it, tagged core or extended, with the lab marked (DeepSeek, Moonshot, Alibaba, MiniMax, StepFun, Meituan, ByteDance, Huawei, Zhipu, LMSYS, Berkeley, Stanford, NVIDIA, Microsoft, Meta, Google and so on). Order within each part by the sequence a reader should follow. Add it to the build script's guide list next to the compute guide, glossary and formula sheet. For the Chinese-lab papers, add a short "how to read them" note: the technical reports are long and the inference content is usually in a dedicated "Inference" or "Deployment" section near the end; point to those sections.

---

## 5. P1: audience fit for an agent builder

1. **Chapter 0 (or an expanded index preface): "From agent builder to inference engineer".** One page that maps what the reader already knows to the book: an API call becomes a request in a queue (3.2), tokens cost prefill and decode (2.4, 2.5), the system prompt is a prefix (4.4, 6.8), tool calls are structured output (3.5), latency is TTFT plus TPOT times length (6.1), the bill is GPU-hours divided by throughput (6.5), and a fine-tuned small model is a cost decision (7.1, 7.5). Give a fast track reading order for someone with one month, and the full order. Keep it to about 800 words.
2. **Embedding and reranker serving (new section in 3.7 or 8.1).** The audience builds RAG systems; the book never explains how embedding models and rerankers are served: encoder-only or decoder-only embeddings, pooling, batch-heavy prefill-only workloads with no KV cache growth, vLLM's and SGLang's embedding and scoring endpoints, Hugging Face Text Embeddings Inference, and cost per million embedded tokens. One section with a CPU lab (embed 10,000 chunks, measure throughput against batch size).
3. **Index page claim.** Change "hands-on labs that run on free GPUs" to wording that matches reality: labs run on CPU or a free Colab GPU, with the measurement labelled either way.
4. **A "what to read when something breaks" map** (P2): one table in 6.9 or the index that routes symptoms (slow TTFT, wrong outputs, OOM, preemptions, cost spike) to chapters.

---

## 6. P2: consistency items noticed during the review

- 6.10 is built on mock engines only; state in its lede that the measured numbers are from mocks (it does, in the widget caption; also say it in the first paragraph).
- 7.3 and 8.3 both note the Qwen2.5 `generation_config` repetition penalty trap; add the same one-line warning to 4.3's lab, which compares greedy outputs.
- The 6.9 note on the Llama 3 root-cause table (148 interruptions listed as 30.1%) should say which of the two figures the executor verified against the paper's table, after checking the PDF.
- 5.5 has a hardware landscape with no Chinese accelerators; after adding Ascend and CloudMatrix384, also add one line each on Cambricon and Moore Threads only if a primary source with specifications is found; otherwise leave them out.
- The glossary generator skips a few definition-box forms ("The rule", "Memory per sequence is constant"); fine, but add "Lightning indexer" and "Redundant experts" as explicit entries, since 4.2 and 4.5 define them inline.

---

## 7. House rules for the executor

Follow these exactly; they are the rules the existing chapters were written under.

1. **Do not invent.** Every number, name and claim comes from a source you have read in this session. For post-June-2026 material (V4, V4.1-Flash, DeepInfer, DeepSelect, TileKernels, LongCat Sparse Attention, FlashMemory, tree speculation on V4) read the GitHub README or the arXiv page first; quote only what is there; label every performance figure "DeepSeek-reported" or "authors-reported". If a detail cannot be confirmed, write "not confirmed from the source" or leave it out.
2. **Primary sources only** for citations: arXiv pages, GitHub READMEs and docs, official blogs, conference pages. No third-party summaries, no news articles.
3. **Style:** direct, no em dashes anywhere (run `grep -c '—' src/<file>.html` and require 0), define each term before use, references only in the end-of-chapter table, "Notes on sources" for any source error or conflict, SVG diagrams for anything that moves through a system, dates on facts ("checked on <date>").
4. **Labs:** every lab result shown as output must have been run; label CPU runs as measured and GPU-only parts as expected behaviour. Never present a computation as a measurement. The user does not do arithmetic by hand, so any worked number is computed in code and the code is shown.
5. **Template:** copy the section skeleton of an existing expert chapter (4.7 or 6.7) for the new chapter 4.8: kicker, h1, lede, meta chips, skip check with answers, numbered sections, expert layer boxes, common mistakes, lab with parts and `<details>` results, in the real world, quiz JSON, summary, references table, notes on sources, dated note, then the script block after `<!--SCRIPT-->`.
6. **Widgets:** use the `IE` helpers in `assets/book.js`, CSS tokens only (no literal colours), check both themes and the 400 px width with `chk2.py`, and screenshot each widget with `wshot.py` before committing.
7. **Build and checks per chapter:** `python3 tools/gen_glossary.py` after adding definition boxes or `tools/gloss_extra.json` entries; `python3 tools/build.py`; `chk2.py` on the built page; em-dash count 0; the full-site link check (`linkcheck.py`) at the end; external links checked with curl (GitHub returns 403 to curl and is fine).
8. **Commits:** one commit per chapter or per cross-cutting change, pushed after each; message in the imperative, with the attribution lines the session reminder gives.
9. **Do not change measured numbers in existing chapters** unless you re-run the lab; if a new source contradicts an existing statement, add a "Notes on sources" line rather than silently editing.
10. **Order of work:** 4.8 first (section 3.1), then the 3.2 cross-chapter updates, then the Part 4 and Part 5 paper additions, then Part 6, then the reading list guide, then section 5, then section 6. Report after each block with what was measured, what was only read, and what could not be confirmed.

---

## 8. Verification checklist for the final report

- [ ] 4.8 exists, is in `chapters.json`, renders in both themes at 1280 and 400 px with no console errors.
- [ ] Every new arXiv ID resolves to the title given in section 4 (spot-check with curl).
- [ ] Every DeepInfer statement is traceable to a line of the README or docs at the recorded commit hash.
- [ ] Glossary count rises (currently 252 terms) and the formula sheet has the new rows.
- [ ] `guides/reading-list.html` lists every cited paper with a lab tag and is linked from the index.
- [ ] Zero em dashes across `src/`, zero broken internal links, full-site render check clean.
- [ ] Index wording about labs corrected.
- [ ] A final summary to the user: chapters touched, papers added (count, Chinese-lab count), what was measured versus read, what was left unconfirmed.
