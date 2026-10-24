# Network weights

[Project overview](../../README.md) | [NNUE architecture](../../docs/nnue.md) | [Engine integration](../../docs/engine.md)

This directory contains five network artifacts spanning three architecture generations. The current engine is compiled for **v3 (big net)** and defaults to **1024_16_32_pairmul_screlu_T60T70Farseer_filtered.bin**.

Filenames, byte counts, SHA-256 checksums, and the compiled default below are verified from this checkout. Architecture-family assignments for historical files follow the existing architecture notes and filenames; the three v3 files have the exact size expected by the current loader. File size alone does not validate tensor contents. Training provenance, strength, and release decisions are intentionally left as **TODO** fields for the owner to fill in. A default is not a claim that a net is the strongest.

## Loading and compatibility

The active dimensions, quantization, and layout are fixed in [network.h](../network.h), [nnue.h](../nnue.h), and [NNUE::load](../nnue.cpp). V3 uses 7,680 input indices, 1,024 values per perspective, pairwise multiplication, hidden widths 16 and 32, and eight material buckets. Its input tensors are int16, dense weights int8, and dense biases int32; `QA/QB/QC = 127/64/64`, output scale 400.

The loader reads 15,867,712 bytes of tensors in order: `l0w`, `l0b`, `l1w`, `l1b`, `l2w`, `l2b`, `l3w`, `l3b`. It rejects undersized files and does not inspect a version header or infer an architecture from the filename. Extra trailing bytes are not interpreted. The smaller v1/v2 files therefore cannot be substituted into the current executable just by changing a path; they require a matching historical evaluator/export layout.

There are currently three distinct path behaviors:

| Mechanism | Current behavior |
|---|---|
| Compiled default | `EngineOptions::nnue_weight_path` in [engine.h](../../include/engine.h) points into `PROJECT_ROOT/nnue/weights/`; the constructor loads it |
| `setoption name nnue_weight_file value <stem>` | Constructs `PROJECT_ROOT/bin/nnue_wgts/<stem>.bin` and immediately calls `nnue.load()`; it does not directly address this directory |
| INI `nnue_weight_path` | `apply_config_file()` updates the stored path relative to `PROJECT_ROOT`, but does not reload NNUE weights |

To change the checkout's startup default, select a compatible v3 file in `EngineOptions` and rebuild. For the existing runtime option, place a compatible file under `bin/nnue_wgts/` and supply its stem without `.bin`. Record the selected file's checksum with any benchmark. The comparison below documents the files in this directory; it does not change the loader or configuration behavior.

## Profile completion guide

Use the same fields for every net so training runs and strength tests can be compared. Replace `TODO` with a value, a source link, or `N/A` with a reason. Do not infer dataset composition, WDL weighting, training duration, or filtering thresholds from filename fragments such as `25wdl`, `1000`, `leela99`, or `filtered`.

For strength results, report the baseline engine **and** baseline net, engine revisions, openings, time control, hardware, thread/hash settings, games and W/D/L (or pentanomial counts), Elo confidence interval, and SPRT configuration/decision where used. Keep raw logs and the test command/config linked alongside the result. Describe changes in both strength and inference/search speed.

## Network comparison

Sizes use decimal MB (1,000,000 bytes). Checksums identify the artifacts currently in this directory. The filenames distinguish the artifacts; dataset names, proportions, and exact filtering recipes remain to be documented and confirmed by the owner.

| Field | v1: Base net | v2: Small net | v3: leela99 | v3: leela99_filt | v3: T60T70Farseer_filtered |
|---|---|---|---|---|---|
| File | [768_128x2.bin](768_128x2.bin) | [output_buckets_25wdl_1000.bin](output_buckets_25wdl_1000.bin) | [1024_16_32_pairmul_screlu_leela99.bin](1024_16_32_pairmul_screlu_leela99.bin) | [1024_16_32_pairmul_screlu_leela99_filt.bin](1024_16_32_pairmul_screlu_leela99_filt.bin) | [1024_16_32_pairmul_screlu_T60T70Farseer_filtered.bin](1024_16_32_pairmul_screlu_T60T70Farseer_filtered.bin) |
| Status | Historical; incompatible with the current v3 loader. | Historical; incompatible with the current v3 loader. | Bundled v3-layout candidate; no strength ranking recorded here. | Bundled v3-layout candidate; no strength ranking recorded here. | Current default in EngineOptions::nnue_weight_path. |
| Architecture | Historical base architecture: 768 inputs, 128 values per perspective, 256-to-1 output. See [v1 layer diagram](../../docs/nnue.md#v1-base-net). | Historical bucketed architecture: 10 input buckets, 1024 values per perspective, SCReLU, 8 output heads. See [v2 layer diagram](../../docs/nnue.md#v2-small-net). | 10 input buckets, dual 1024-wide accumulators, pairwise multiplication, 16/32 hidden layers, 8 output heads. See [v3 layer diagram](../../docs/nnue.md#v3-big-net). | 10 input buckets, dual 1024-wide accumulators, pairwise multiplication, 16/32 hidden layers, 8 output heads. See [v3 layer diagram](../../docs/nnue.md#v3-big-net). | 10 input buckets, dual 1024-wide accumulators, pairwise multiplication, 16/32 hidden layers, 8 output heads. See [v3 layer diagram](../../docs/nnue.md#v3-big-net). |
| Size (bytes) | 197,440 | 15,763,520 | 15,867,712 | 15,867,712 | 15,867,712 |
| Size (MB) | 0.197 | 15.764 | 15.868 | 15.868 | 15.868 |
| SHA-256 | `63aea5ec78407a9c496dd05b487d6d166d00e227c335dd5088da62a0e70e2c23` | `aa55c5deafeb180fb5f90c9138611529afe2eead40bda00d8ecbc091113c2085` | `f8b31563095399e31ad7388d954fc5038122e4056c440be454e7dd2b280d68d9` | `40fb648096394a3bc9b105d7ff7e654c0816f7a701b6071f389d1b0ee997e8f1` | `cb2b86bedc23d68ae8d036b828ec8d8ed94c4befd1a21f623c0ff8edbfd4514b` |
| **Identity and intended use** | | | | | |
| Author / maintainer | TODO | TODO | TODO | TODO | TODO |
| Created / exported date | TODO | TODO | TODO | TODO | TODO |
| Training run ID / checkpoint / epoch | TODO | TODO | TODO | TODO | TODO |
| Engine release or first compatible revision | TODO | TODO | TODO | TODO | TODO |
| Purpose and expected strengths | TODO | TODO | TODO | TODO | TODO |
| Recommended use / selection rationale | TODO | TODO | TODO | TODO | TODO |
| Known weaknesses / limitations | TODO | TODO | TODO | TODO | TODO |
| Artifact and dataset licenses / attribution | TODO | TODO | TODO | TODO | TODO |
| **Training data and labels** | | | | | |
| Sources / dataset versions / download or source links | TODO | TODO | TODO | TODO | TODO |
| Mixture proportions and sampling weights | TODO | TODO | TODO | TODO | TODO |
| Raw positions / retained positions / unique positions | TODO | TODO | TODO | TODO | TODO |
| Filtering pipeline and exact thresholds | TODO | TODO | TODO | TODO | TODO |
| Deduplication and train/validation/test split | TODO | TODO | TODO | TODO | TODO |
| Position coverage: openings, endgames, material, checks | TODO | TODO | TODO | TODO | TODO |
| Label source and teacher engine/net revision | TODO | TODO | TODO | TODO | TODO |
| Teacher search settings / evaluation perspective / scaling | TODO | TODO | TODO | TODO | TODO |
| WDL and score target blend / loss / target transforms | TODO | TODO | TODO | TODO | TODO |
| **Training and export** | | | | | |
| Trainer repository / commit / config / command | TODO | TODO | TODO | TODO | TODO |
| Initialization: random seed, parent net, fine-tuning | TODO | TODO | TODO | TODO | TODO |
| Optimizer / learning-rate schedule / regularization | TODO | TODO | TODO | TODO | TODO |
| Batch size / batches / epochs / positions seen | TODO | TODO | TODO | TODO | TODO |
| Hardware / runtime / precision | TODO | TODO | TODO | TODO | TODO |
| Validation metrics and checkpoint-selection rule | TODO | TODO | TODO | TODO | TODO |
| Factoriser / quantization-aware training / clipping | TODO | TODO | TODO | TODO | TODO |
| Exporter revision / tensor order / quantization / format | TODO | TODO | TODO | TODO | TODO |
| Export verification: scalar vs SIMD, incremental vs full | TODO | TODO | TODO | TODO | TODO |
| **Evaluation and release decision** | | | | | |
| Baseline engine revision and baseline net SHA-256 | TODO | TODO | TODO | TODO | TODO |
| Candidate engine revision / compiler / build flags | TODO | TODO | TODO | TODO | TODO |
| Test framework revision / command / configuration | TODO | TODO | TODO | TODO | TODO |
| Hardware / threads / hash / tablebases / adjudication | TODO | TODO | TODO | TODO | TODO |
| Openings / colors / time control / game count | TODO | TODO | TODO | TODO | TODO |
| W/D/L or pentanomial results | TODO | TODO | TODO | TODO | TODO |
| Elo estimate / confidence interval / statistical method | TODO | TODO | TODO | TODO | TODO |
| SPRT hypotheses / error rates / LLR / decision | TODO | TODO | TODO | TODO | TODO |
| Inference speed / search NPS / memory / fixed-node tests | TODO | TODO | TODO | TODO | TODO |
| Tactical or positional suite results / regressions | TODO | TODO | TODO | TODO | TODO |
| Raw logs / report links / reproducibility notes | TODO | TODO | TODO | TODO | TODO |
| Accepted, experimental, superseded, or rejected; reason | TODO | TODO | TODO | TODO | TODO |
