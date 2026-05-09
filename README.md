# Numerical Fact Checking Using SLM


A two-stage Retrieval-Augmented Generation (RAG) pipeline for verifying real-world numerical and temporal claims. Given a claim, the system retrieves relevant evidence from a 426,741-document corpus and classifies it as **True**, **False**, or **Conflicting** — using only a 7B small language model with no task-specific fine-tuning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Pipeline Architecture](#pipeline-architecture)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Example](#example)
- [Citation](#citation)

---

## Overview

Numerical claims are uniquely difficult to verify. A transposition of two digits — "8.4%" vs "4.8%" — renders an otherwise plausible claim factually wrong, yet standard semantic similarity pipelines cannot catch this. This project addresses that challenge through:

1. **Claim Decomposition** — A large language model breaks each claim into 5 atomic yes/no sub-questions, broadening the retrieval surface while preserving all numerical and temporal values verbatim.
2. **BM25 Sparse Retrieval** — Top-100 candidate documents are retrieved from the evidence corpus for each claim.
3. **Dense Semantic Reranking** — BAAI/bge-m3 (568M params) re-scores candidates using 5 query vectors (original claim ⊕ each sub-question), producing a 5×100 cosine similarity matrix.
4. **Top-1 Evidence Selection** — The best-scoring unique document per sub-question is selected (up to 5 deduplicated evidences).
5. **Veracity Prediction** — Qwen-2.5-7B-Instruct performs zero-shot chain-of-thought reasoning over the evidence to output a final label + justification.

---

## Results

### Leaderboard — CLEF 2025 CheckThat! Task 3 (English)

| Rank | Team | Macro-F1 | True-F1 | False-F1 | Conf-F1 |
|------|------|----------|---------|---------|---------|
| 1 | LIS | 0.5954 | 0.633 | 0.828 | 0.325 |
| 2 | TIFIN | 0.5570 | — | — | — |
| 3 | DS@GT | 0.5210 | 0.550 | 0.810 | 0.360 |
| 4 | Fraunhofer SIT | 0.5213 | 0.449 | 0.766 | 0.350 |
| 5 | JU_NLP | 0.4883 | — | — | — |
| 6 | CornellNLP | 0.4857 | 0.509 | 0.782 | 0.166 |
| **7 ⭐** | **Ours** | **0.4749** | **0.509** | **0.750** | **0.166** |
| 9 | ClaimIQ (8B + LoRA) | 0.4300 | 0.420 | 0.760 | 0.110 |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| True | 0.4948 | 0.5241 | 0.5090 | 540 |
| Conflicting | 0.3009 | 0.1147 | 0.1661 | 593 |
| False | 0.6956 | 0.8127 | 0.7496 | 1,991 |
| **Macro Avg** | **0.4971** | **0.4838** | **0.4749** | 3,124 |

### Model Scale vs. Performance

| System | Params | Fine-Tuned? | Macro-F1 | Rank |
|--------|--------|------------|---------|------|
| LIS | ~63B | Yes (LoRA) | 0.5954 | 1 |
| CornellNLP | ~182B | LLM Only | 0.4857 | 6 |
| **Ours** | **7B** | **None** | **0.4749** | **7** |
| ClaimIQ | 8B | Yes (LoRA) | 0.4300 | 9 |

Our 7B zero-shot system **outperforms a fine-tuned 8B competitor (ClaimIQ) by 4.49 F1 points**, demonstrating that prompt engineering and retrieval quality can substitute for fine-tuning.

---

## Pipeline Architecture

```
Input Claim
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 1: EVIDENCE RETRIEVAL        │
│                                     │
│  Claim → LLM Decomposer             │
│          (Gemini-2.5-Flash /        │
│           Gemini-1.5-Flash /        │
│           Qwen-2.5-7B-Instruct)     │
│               │                     │
│               ▼                     │
│         5 Sub-questions             │
│               │                     │
│               ▼                     │
│  BM25 Retrieval (Top-100 docs)      │
│               │                     │
│               ▼                     │
│  BAAI/bge-m3 Dense Reranking        │
│  (5 × 100 cosine similarity matrix) │
│               │                     │
│               ▼                     │
│  Top-1 per sub-question → dedupe    │
│  → up to 5 unique evidence docs     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 2: VERACITY PREDICTION       │
│                                     │
│  Claim + Sub-questions + Evidence   │
│               │                     │
│               ▼                     │
│  Qwen-2.5-7B-Instruct (zero-shot)  │
│  Chain-of-thought structured prompt │
│               │                     │
│               ▼                     │
│  {"label": "True/False/Conflicting",│
│   "reason": "..."}                  │
└─────────────────────────────────────┘
```

---

## Repository Structure

```
Numerical-Fact-Checking/
│
├── claim_decomposition.ipynb       # Stage 1a: LLM-based claim → 5 sub-questions
│                                   #   Uses Gemini-2.5-Flash / Gemini-1.5-Flash /
│                                   #   Qwen-2.5-7B-Instruct with JSON output prompt
│
├── BM25.ipynb                      # Stage 1b: BM25 sparse retrieval
│                                   #   Retrieves top-100 docs per claim from the
│                                   #   426,741-document QuanTemp evidence corpus
│
├── downloading-embeding-model.ipynb # Downloads & caches BAAI/bge-m3 locally
│
├── concatenation.ipynb             # Builds query strings: claim ⊕ sub-question
│                                   #   (5 query vectors per claim for reranking)
│
├── reranking.ipynb                 # Stage 1c: Dense reranking with bge-m3
│                                   #   Embeds queries + docs, computes 5×100
│                                   #   cosine similarity matrix
│
├── evidence-retrival.ipynb         # Stage 1d: Top-1 evidence selection + dedup
│                                   #   Selects best doc per sub-question,
│                                   #   deduplicates across all 5 selections
│
├── processing_reranked.ipynb       # Post-processing: formats reranked evidence
│                                   #   into structured JSON for the predictor
│
├── predict_veracity_ollama.py      # Stage 2: Veracity prediction (v1)
│                                   #   Qwen-2.5-7B via Ollama, zero-shot CoT
│
├── predict_veracity_ollama-new.py  # Stage 2: Veracity prediction (v2, improved)
│                                   #   Updated prompt with stricter numerical rules
│
└── results.ipynb                   # Evaluation: confusion matrix, per-class F1,
                                    #   leaderboard comparison
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: 16GB+ VRAM for bge-m3 + Qwen-2.5-7B)
- [Ollama](https://ollama.com) installed locally (for Qwen-2.5-7B inference)

### 1. Clone the repository

```bash
git clone https://github.com/medhairya/Numerical-Fact-Checking.git
cd Numerical-Fact-Checking
```

### 2. Install dependencies

```bash
pip install torch transformers sentence-transformers rank_bm25 \
            google-generativeai jupyter pandas numpy scikit-learn \
            tqdm ollama
```

### 3. Pull the Qwen model via Ollama

```bash
ollama pull qwen2.5:7b
```

### 4. Set up API keys

For Gemini-based claim decomposition, set your API key:

```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
```

---

## Usage

Run each notebook **in order** — each stage produces output files consumed by the next.

### Step 1 — Decompose claims into sub-questions

```bash
jupyter notebook claim_decomposition.ipynb
```

**Input:** `claims.json` (list of `{claim_id, claim_text}`)  
**Output:** `decomposed_questions.json` (5 yes/no sub-questions per claim)

---

### Step 2 — BM25 sparse retrieval

```bash
jupyter notebook BM25.ipynb
```

**Input:** `claims.json`, evidence corpus  
**Output:** `bm25_top100.json` (top-100 retrieved doc IDs per claim)

---

### Step 3 — Download the embedding model

```bash
jupyter notebook downloading-embeding-model.ipynb
```

Downloads and caches `BAAI/bge-m3` locally for offline inference.

---

### Step 4 — Build concatenated query strings

```bash
jupyter notebook concatenation.ipynb
```

**Input:** `decomposed_questions.json`  
**Output:** `query_strings.json` (5 `claim ⊕ sub-question` strings per claim)

---

### Step 5 — Dense reranking

```bash
jupyter notebook reranking.ipynb
```

**Input:** `query_strings.json`, `bm25_top100.json`  
**Output:** `reranked_scores.json` (5×100 cosine similarity matrix per claim)

---

### Step 6 — Evidence selection

```bash
jupyter notebook evidence-retrival.ipynb
```

**Input:** `reranked_scores.json`  
**Output:** `top_evidences.json` (up to 5 unique, highest-scoring docs per claim)

---

### Step 7 — Post-process reranked evidence

```bash
jupyter notebook processing_reranked.ipynb
```

**Input:** `top_evidences.json`  
**Output:** `prediction_input.json` (formatted for the veracity predictor)

---

### Step 8 — Predict veracity

```bash
python predict_veracity_ollama-new.py
```

**Input:** `prediction_input.json`  
**Output:** `predictions.json` — one label + reason per claim

```json
{
  "claim_id": 14,
  "label": "True",
  "reason": "Evidence confirms 1,100 suicides from 2005–2009 at one per 36 hours."
}
```

---

### Step 9 — Evaluate results

```bash
jupyter notebook results.ipynb
```

Generates: confusion matrix, per-class precision/recall/F1, macro-F1 score.

---

## Dataset

This project uses the **QuanTemp** benchmark from CLEF 2025 CheckThat! Lab Task 3.

| Split | Claims | True | False | Conflicting |
|-------|--------|------|-------|-------------|
| Train | 9,935 | ~18% | ~58% | ~24% |
| Validation | 3,084 | ~18% | ~58% | ~24% |
| Test | 3,656 | 540 | 1,991 | 593 |

**Evidence corpus:** 426,741 documents

The dataset is available through the official [CLEF 2025 CheckThat! Lab](https://checkthat.gitlab.io/clef2025/). You will need to register and download it separately — it cannot be redistributed here.

---

## Example

**Claim:** *"From '05 to '09, we've had 1,100 soldiers commit suicide, one every 36 hours."*

**Generated Sub-questions:**
1. Were there 1,100 soldier suicides between 2005 and 2009?
2. Did soldier suicides average one every 36 hours during this period?
3. Were the suicides exclusive to a specific military branch?
4. Did the data originate from a credible military or government source?
5. Was this claim documented in an official military report?

**Top Retrieved Evidence** *(cosine score: 0.87)*:  
> *"The Army reported 1,100 active-duty and veteran suicides between fiscal years 2005–2009, averaging approximately one suicide every 36 hours, according to data released by the Department of Defense in 2010."*

**Verdict:** ✅ **True** — Evidence explicitly confirms both the count (1,100) and the rate (one per 36 hours) over the stated period.

---

## Models Used

| Model | Role | Params |
|-------|------|--------|
| Gemini-2.5-Flash | Primary claim decomposer | — (API) |
| Gemini-1.5-Flash-preview | Fallback decomposer | — (API) |
| Qwen-2.5-7B-Instruct | Decomposer (offline) + Veracity predictor | 7B |
| BAAI/bge-m3 | Dense bi-encoder for reranking | 568M |

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{patel2025numerical,
  title     = {Numerical Fact Checking Using SLM},
  author    = {Patel, Dhairya},
  booktitle = {Working Notes of CLEF 2025 CheckThat! Lab},
  year      = {2025},
  address   = {Madrid, Spain}
}
```

This work was evaluated on the QuanTemp benchmark:

```bibtex
@inproceedings{venktesh2024quantemp,
  title     = {QuanTemp: A Real-World Open-Domain Benchmark for Fact-Checking Numerical Claims},
  author    = {Venktesh, V. and others},
  booktitle = {Proc. ACM SIGIR},
  year      = {2024}
}
```

---

## Author

**Dhairya Patel**  
MTech ICT – Machine Learning  
Dhirubhai Ambani University, Gandhinagar

---

<p align="center">
  <sub>Ranked 7th out of 83 teams · CLEF 2025 CheckThat! Lab Task 3 · Macro-F1: 0.4749</sub>
</p>
