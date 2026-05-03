---
title: "Assignment 3 Experiment Documentation"
subtitle: "IterAlign PEFT/LoRA Extension"
geometry: margin=1in
fontsize: 11pt
header-includes:
  - \usepackage{graphicx}
---

# Overview

This document records the experiment package for Assignment 3. The main report
describes the research narrative; this document captures the practical setup:
methods, datasets, configuration, commands, metrics, result artifacts, and
reproducibility notes.

The extension replaces the full fine-tuning step in the reproduced IterAlign
pipeline with PEFT/LoRA adapter tuning. DangerousQA is the selected additional
dataset for the Assignment 3 expansion requirement.

# Research Question

Can LoRA-based parameter-efficient fine-tuning preserve most of the alignment
gains of the reproduced IterAlign baseline while reducing trainable parameters
and training time?

# Methods

| Method | Description | Status |
|---|---|---|
| Vanilla model | Unaligned small model before IterAlign | Assignment 2 baseline |
| Full FT IterAlign | IterAlign with full supervised fine-tuning | Assignment 2 baseline |
| LoRA IterAlign | PEFT/LoRA extension of IterAlign | Assignment 3 method |

# Datasets

| Dataset | Role | Status |
|---|---|---|
| hh-rlhf red-team pool | Red-team / alignment prompts | Existing reproduction dataset |
| HarmfulQA | Harmfulness evaluation | Existing reproduction dataset |
| DangerousQA | Dangerous-instruction evaluation | Assignment 3 additional dataset |

# Configuration

| Field | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolLM-135M` |
| Oracle / judge model | Gemini 3.1 Flash Lite preview |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | Auto-detected `q_proj`, `k_proj`, `v_proj`, `o_proj` projection layers |
| LoRA trainable parameters | Approximately 1.84M adapter parameters, computed from SmolLM-135M attention dimensions and rank 16 |
| Learning rate | 2e-6 |
| Epochs | 3 |
| Per-device batch size | 2 |
| Gradient accumulation | 4 |
| Max sequence length | 512 |
| Dataset sizes / caps | hh-rlhf capped at 1000; HarmfulQA 1960; DangerousQA 200 |
| Random seed | Not reported in the available artifacts |

# Commands

Baseline / original full-fine-tuning code path:

```bash
python constitution_induced_inference_iterated_llama2.py
```

LoRA / PEFT IterAlign run:

```bash
python iteralign_smollm_lora.py
```

Evaluation:

```text
Scores are taken from the supplied Gemini-judge artifacts and benchmark figure.
```

# Metrics

Primary quality metric:

- Gemini-judge score.

Efficiency metrics:

- Trainable parameter count.
- Total training time.
- Estimated peak VRAM.

Statistical note:

- The available artifact contains one run per method and dataset.
- The reported comparison is descriptive rather than a statistical significance claim.

# Results

| Method | Dataset | Gemini score | Trainable params | Training time |
|---|---|---:|---:|---:|
| Vanilla | hh-rlhf | 0.23 | N/A | N/A |
| Full FT | hh-rlhf | 0.42 | All base-model parameters | ~7 days |
| LoRA | hh-rlhf | 0.38 | ~1.84M adapter parameters | ~4 days |
| Full FT | HarmfulQA | 0.45 | All base-model parameters | ~7 days |
| LoRA | HarmfulQA | 0.41 | ~1.84M adapter parameters | ~4 days |
| Full FT | DangerousQA | 0.46 | All base-model parameters | ~7 days |
| LoRA | DangerousQA | 0.42 | ~1.84M adapter parameters | ~4 days |

| Mode | Estimated peak VRAM |
|---|---:|
| Full fine-tuning | about 2.5-3.5 GB |
| LoRA fine-tuning | about 0.9-1.5 GB |

# Figure Artifact

\begin{center}
\includegraphics[width=0.95\linewidth]{figures/benchplot.png}

\small Training efficiency comparison across datasets.
\end{center}

Observed from the figure:

- Vanilla baseline is 0.23.
- Full precision reaches 0.42, 0.45, and 0.46 after roughly 7 days.
- PEFT+LoRA reaches 0.38, 0.41, and 0.42 after roughly 4 days.
- The score gap is 0.04 on each dataset, while training time drops by roughly 3 days.

# Output Paths

LoRA runs write artifacts under:

```text
output_smollm_lora_<dataset>/
```

Expected files inside each output directory:

- `training_state.json`
- `constitution_batch_<batch_id>.txt`
- `neg_prompts_batch_<batch_id>.pkl`
- `SFT_data_batch_<batch_id>.json`
- `sft_batch_<batch_id>/`


# Reproducibility Notes

- The Gemini judge requires API access, so exact reruns depend on the configured API key and endpoint (and model) availability.
- Dataset sizes are capped where noted above.
- Since only one run is available per method and dataset, the results should be described as descriptive rather than statistically significant.
