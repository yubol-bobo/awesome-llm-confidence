# Awesome LLM Confidence Utilizations

[![Website](https://img.shields.io/badge/🌐_Project_Page-Visit-blue)](https://yubol-bobo.github.io/awesome-llm-confidence/)
[![Paper](https://img.shields.io/badge/📄_Paper-Preprint-green)](https://yubol-bobo.github.io/assets/pdf/Conf_Survey.pdf)
[![Awesome](https://img.shields.io/badge/Awesome-LLM_Confidence-orange)](https://github.com/yubol-bobo/awesome-llm-confidence)

> **Confidence as Control: A Survey of Confidence Utilization in Large Language Models**
>
> [Yubo Li](mailto:yubol@andrew.cmu.edu), Tianyang Zhou, Xiaobin Shen, Yidi Miao, Rema Padman, Ramayya Krishnan
>
> Carnegie Mellon University | 2026

## 📖 Overview

Most work on confidence in large language models has focused on **estimation**, uncertainty quantification, and calibration. In deployed systems, however, the key question is how confidence should be **used to govern behavior**.

This survey studies **confidence utilization**: the use of confidence-related signals to control system decisions. We formalize this through a unified framework in which confidence is defined over decision units under a local state and then consumed by a policy to determine actions.

<p align="center">
  <img src="assets/img/diagram.png" width="600" alt="Taxonomy">
</p>

---

## 📑 Table of Contents

- [Unified Framework](#-unified-framework)
- [§3 Confidence-Aware Training](#3-confidence-aware-training-28-papers)
  - [Data Selection](#data-selection)
  - [Fine-Tuning & Distillation](#fine-tuning--distillation)
  - [Preference Optimization & RL](#preference-optimization--rl)
- [§4 Confidence-Driven Inference](#4-confidence-driven-inference-19-papers)
  - [Output Selection](#output-selection)
  - [Adaptive Stopping & Search](#adaptive-stopping--search)
  - [Decoding Control](#decoding-control)
- [§5 Confidence-Guided Model Selection](#5-confidence-guided-model-selection-14-papers)
  - [Sequential Cascading](#sequential-cascading)
  - [Pre-Call Routing](#pre-call-routing)
  - [Hybrid Systems](#hybrid-systems)
- [§6 Confidence-Gated RAG](#6-confidence-gated-rag-22-papers)
  - [When to Retrieve](#when-to-retrieve)
  - [What Context to Keep](#what-context-to-keep)
  - [Groundedness Detection](#groundedness-detection)
  - [Abstention & Conformal](#abstention--conformal)
- [§7 Confidence-Based Risk Management](#7-confidence-based-risk-management-24-papers)
  - [Actionable Signals](#actionable-signals)
  - [Hallucination Detection](#hallucination-detection)
  - [Conformal Guarantees](#conformal-guarantees)
  - [Abstention & Alignment](#abstention--alignment)
- [§8 Confidence in Agentic Systems](#8-confidence-in-agentic-systems-14-papers)
  - [Selective Escalation](#selective-escalation)
  - [Self-Correction](#self-correction)
  - [Verifier-Guided Search](#verifier-guided-search)
  - [Multi-Agent Deliberation](#multi-agent-deliberation)
- [Open Challenges](#-open-challenges)
- [Citation](#-citation)

---

## 🔬 Unified Framework

Every method in this survey follows the same abstraction:

```
Decision State (ξ_t) + Decision Units (U_t)
    → Score κ_t(u; ξ_t)
    → Transform T_t(κ_t)
    → Policy δ_t → Action a_t
    → Updated State ξ_{t+1}
```

Methods are characterized along **three axes**:

| Axis | Categories |
|------|-----------|
| **Source** | Self, Sample-based, Auxiliary, External, Hybrid |
| **Unit / Granularity** | Token, Claim, Candidate, Model, Step, Trajectory, Episode |
| **Functional Role** | Selection, Weighting, Allocation, Control-flow, Aggregation, Learning signal |

---

## §3 Confidence-Aware Training (28 papers)

Confidence determines where learning should occur, be damped, or teach the model to abstain.

### Data Selection

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning](https://arxiv.org/abs/2308.12032) | Li et al. | NAACL | 2024 |
| [Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning](https://arxiv.org/abs/2402.00530) | Li et al. | Preprint | 2024 |
| [Automatic Instruction Data Selection for Large Language Models via Uncertainty-Aware Influence Maximization](https://dl.acm.org/doi/10.1145/3696410.3714817) | Han et al. | WWW | 2025 |
| [Active Preference Learning for Large Language Models](https://arxiv.org/abs/2402.08114) | Muldrew et al. | Preprint | 2024 |
| [Selectit: Selective Instruction Tuning for Large Language Models via Uncertainty-Aware Self-Reflection](https://arxiv.org/abs/2402.16705) | Liu et al. | Preprint | 2024 |
| [Automated Data Curation for Robust Language Model Fine-Tuning](https://arxiv.org/abs/2403.12776) | Chen & Mueller | Preprint | 2024 |
| [How to Train Data-Efficient LLMs](https://arxiv.org/abs/2402.09668) | Sachdeva et al. | Preprint | 2024 |

### Fine-Tuning & Distillation

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Enhancing Trust in Large Language Models with Uncertainty-Aware Fine-Tuning](https://arxiv.org/abs/2412.02904) | Krishnan et al. | Preprint | 2024 |
| [Know the Unknown: An Uncertainty-Sensitive Method for LLM Instruction Tuning](https://arxiv.org/abs/2406.10099) | Li et al. | ACL Findings | 2025 |
| [C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](https://arxiv.org/abs/2505.17773) | Rahmati et al. | Preprint | 2025 |
| [SelecTKD: Selective Token-Weighted Knowledge Distillation for LLMs](https://arxiv.org/abs/2510.24021) | Huang et al. | Preprint | 2025 |
| [Revisiting Knowledge Distillation for Autoregressive Language Models](https://arxiv.org/abs/2402.11890) | Zhong et al. | Preprint | 2024 |
| [Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning](https://aclanthology.org/2024.findings-acl.958/) | Li et al. | ACL Findings | 2024 |

### Preference Optimization & RL

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | Rafailov et al. | NeurIPS | 2023 |
| [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) | Meng et al. | NeurIPS | 2024 |
| [β-DPO: Direct Preference Optimization with Dynamic β](https://arxiv.org/abs/2407.08639) | Wu et al. | NeurIPS | 2024 |
| [ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](https://arxiv.org/abs/2506.08712) | Yoon et al. | ICML | 2025 |
| [CAPO: Confidence Aware Preference Optimization Learning for Multilingual Preferences](https://arxiv.org/abs/2511.07691) | Pokharel et al. | Preprint | 2025 |
| [Enhancing LLM Reasoning via Non-Human-Like Reasoning Path Preference Optimization](https://arxiv.org/abs/2510.11104) | Lu et al. | Preprint | 2025 |
| [Confidence as a Reward: Transforming LLMs into Reward Models](https://arxiv.org/abs/2510.13501) | Du et al. | Preprint | 2025 |
| [Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models](https://arxiv.org/abs/2506.06395) | Li et al. | Preprint | 2025 |
| [Maximizing Confidence Alone Improves Reasoning](https://arxiv.org/abs/2505.22660) | Prabhudesai et al. | Preprint | 2025 |
| [CoDaPo: Confidence and Difficulty-Adaptive Policy Optimization for Post-Training Language Models](https://openreview.net/forum?id=O9CYgZFtm7) | Zhou et al. | ICML Workshop | 2025 |
| [C²GSPG: Calibration-Aware Sequence RL](https://arxiv.org/abs/2509.23129) | Liu et al. | Preprint | 2025 |
| [Uncertainty-Penalized Reinforcement Learning from Human Feedback with Diverse Reward LoRA Ensembles](https://arxiv.org/abs/2401.00243) | Zhai et al. | Preprint | 2024 |
| [Towards Reliable Alignment: Uncertainty-Aware RLHF](https://arxiv.org/abs/2410.23726) | Banerjee & Gopalan | Preprint | 2024 |
| [Taming Overconfidence in LLMs: Reward Calibration in RLHF](https://arxiv.org/abs/2410.09724) | Leng et al. | Preprint | 2024 |
| [Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning](https://arxiv.org/abs/2512.19920) | Wu et al. | Preprint | 2025 |

---

## §4 Confidence-Driven Inference (19 papers)

Confidence becomes an online control variable at candidate, state, and token levels.

### Output Selection

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) | Wang et al. | ICLR | 2023 |
| [Confidence Improves Self-Consistency in LLMs](https://arxiv.org/abs/2502.06233) | Taubenfeld et al. | ACL Findings | 2025 |
| [Universal Self-Consistency for Large Language Model Generation](https://arxiv.org/abs/2311.17311) | Chen et al. | Preprint | 2023 |
| [ACR: Adaptive Confidence Re-Scoring for Reliable Answer Selection Among Multiple Candidates](https://arxiv.org/abs/2505.20393) | Jeong & Choi | Preprint | 2025 |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | Lightman et al. | ICLR | 2024 |
| [Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations](https://arxiv.org/abs/2312.08935) | Wang et al. | ACL | 2024 |
| [SteerConf: Steering LLMs for Confidence Elicitation](https://arxiv.org/abs/2503.02863) | Zhou et al. | NeurIPS | 2025 |

### Adaptive Stopping & Search

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs](https://arxiv.org/abs/2305.11860) | Aggarwal et al. | EMNLP | 2023 |
| [Efficient test-time scaling via self-calibration](https://arxiv.org/abs/2503.00031) | Huang et al. | Preprint | 2025 |
| [Deep think with confidence](https://arxiv.org/abs/2508.15260) | Fu et al. | Preprint | 2025 |
| [Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions](https://arxiv.org/abs/2503.22353) | Li et al. | ACL Findings | 2025 |
| [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) | Yao et al. | NeurIPS | 2023 |
| [Concise: Confidence-guided compression in step-by-step efficient reasoning](https://arxiv.org/abs/2505.04881) | Qiao et al. | EMNLP | 2025 |

### Decoding Control

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Contrastive decoding: Open-ended text generation as optimization](https://arxiv.org/abs/2210.15097) | Li et al. | ACL | 2023 |
| [DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/abs/2309.03883) | Chuang et al. | ICLR | 2024 |
| [Delta--Contrastive Decoding Mitigates Text Hallucinations in Large Language Models](https://arxiv.org/abs/2502.05825) | Huang & Chen | Preprint | 2025 |
| [COCOA: Confidence-and Context-Aware Adaptive Decoding for Resolving Knowledge Conflicts in Large Language Models](https://arxiv.org/abs/2508.17670) | Khandelwal et al. | EMNLP | 2025 |
| [Active layer-contrastive decoding reduces hallucination in large language model generation](https://arxiv.org/abs/2505.23657) | Zhang et al. | EMNLP | 2025 |
| [Confidence-aware sub-structure beam search (cabs): Mitigating hallucination in structured data generation with large language models](https://arxiv.org/abs/2406.00069) | Wei et al. | Preprint | 2024 |

---

## §5 Confidence-Guided Model Selection (14 papers)

Confidence governs which model to call, when to defer, and how to allocate capacity across portfolios.

### Sequential Cascading

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Language Model Cascades: Token-Level Uncertainty and Beyond](https://arxiv.org/abs/2404.10136) | Gupta et al. | Preprint | 2024 |
| [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176) | Chen et al. | Preprint | 2023 |
| [AutoMix: Automatically Mixing Language Models](https://arxiv.org/abs/2310.12963) | Madaan et al. | NeurIPS | 2024 |
| [Gatekeeper: Improving Model Cascades Through Confidence Tuning](https://arxiv.org/abs/2502.19335) | Rabanser et al. | NeurIPS | 2025 |
| [Rational Tuning of LLM Cascades via Probabilistic Modeling](https://arxiv.org/abs/2501.09345) | Zellinger & Thomson | TMLR | 2025 |
| [Faster Cascades via Speculative Decoding](https://arxiv.org/abs/2405.19261) | Narasimhan et al. | Preprint | 2024 |

### Pre-Call Routing

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [OptLLM: Optimal Assignment of Queries to Large Language Models](https://arxiv.org/abs/2405.15130) | Liu et al. | ICWS | 2024 |
| [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665) | Ong et al. | Preprint | 2024 |
| [Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing](https://arxiv.org/abs/2404.14618) | Ding et al. | ICLR | 2024 |
| [CARGO: A Framework for Confidence-Aware Routing of Large Language Models](https://arxiv.org/abs/2509.14899) | Barrak et al. | Preprint | 2025 |
| [Leveraging Uncertainty Estimation for Efficient LLM Routing](https://arxiv.org/abs/2502.11021) | Zhang et al. | Preprint | 2025 |
| [Learning to Route LLMs with Confidence Tokens](https://arxiv.org/abs/2410.13284) | Chuang et al. | ICML | 2025 |

### Hybrid Systems

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [A Unified Approach to Routing and Cascading for LLMs](https://arxiv.org/abs/2410.10347) | Dekoninck et al. | Preprint | 2024 |
| [Select-then-Route: Taxonomy Guided Routing for LLMs](https://aclanthology.org/2025.emnlp-industry.28/) | Shah & Shridhar | EMNLP Industry | 2025 |

---

## §6 Confidence-Gated RAG (22 papers)

Confidence becomes source-sensitive across parametric and non-parametric knowledge.

### When to Retrieve

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983) | Jiang et al. | EMNLP | 2023 |
| [DRAGIN: Dynamic Retrieval Augmented Generation Based on the Information Needs of Large Language Models](https://arxiv.org/abs/2403.10081) | Su et al. | Preprint | 2024 |
| [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models Through Question Complexity](https://arxiv.org/abs/2403.14403) | Jeong et al. | Preprint | 2024 |
| [Self-Knowledge Guided Retrieval Augmentation for Large Language Models](https://arxiv.org/abs/2310.05002) | Wang et al. | Preprint | 2023 |
| [SELF-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection](https://arxiv.org/abs/2310.11511) | Asai et al. | ICLR (Oral) | 2024 |
| [SEAKR: Self-Aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](https://arxiv.org/abs/2406.19215) | Yao et al. | ACL | 2025 |
| [SUGAR: Leveraging Contextual Confidence for Smarter Retrieval](https://arxiv.org/abs/2501.04899) | Zubkova et al. | ICASSP | 2025 |
| [PAIRS: Parametric-Verified Adaptive Information Retrieval and Selection for Efficient RAG](https://arxiv.org/abs/2508.04057) | Chen et al. | Preprint | 2025 |

### What Context to Keep

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) | Yan et al. | Preprint | 2024 |
| [Learning to Filter Context for Retrieval-Augmented Generation](https://arxiv.org/abs/2311.08377) | Wang et al. | Preprint | 2023 |
| [InfoGain-RAG: Boosting Retrieval-Augmented Generation Through Document Information Gain-Based Reranking and Filtering](https://arxiv.org/abs/2509.12765) | Wang et al. | EMNLP | 2025 |
| [SKILL-RAG: Self-Knowledge Induced Learning and Filtering for Retrieval-Augmented Generation](https://arxiv.org/abs/2509.20377) | Isoda | Preprint | 2025 |
| [Sparse-RAG: Sparse Document Selection](https://arxiv.org/abs/2405.16178) | Zhu et al. | Preprint | 2024 |
| [UncertaintyRAG: Span-Level Uncertainty Enhanced Long-Context Modeling for Retrieval-Augmented Generation](https://arxiv.org/abs/2410.02719) | Li et al. | Preprint | 2024 |

### Groundedness Detection

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability](https://arxiv.org/abs/2410.11414) | Sun et al. | Preprint | 2024 |
| [HALT-RAG: A Task-Adaptable Framework for Hallucination Detection with Calibrated NLI Ensembles and Abstention](https://arxiv.org/abs/2509.07475) | Goswami & Kurra | Preprint | 2025 |
| [Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation](https://arxiv.org/abs/2505.21072) | Fadeeva et al. | Preprint | 2025 |

### Abstention & Conformal

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction](https://arxiv.org/abs/2307.04642) | Li et al. | NAACL | 2024 |
| [ConFLARE: Conformal Large Language Model Retrieval](https://arxiv.org/abs/2404.04287) | Rouzrokh et al. | Preprint | 2024 |
| [Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction](https://arxiv.org/abs/2511.17908) | Chakraborty et al. | Preprint | 2025 |
| [Response Quality Assessment for Retrieval-Augmented Generation via Conditional Conformal Factuality](https://arxiv.org/abs/2506.20978) | Feng et al. | SIGIR | 2025 |
| [Divide-Then-Align: Honest Alignment Based on the Knowledge Boundary of RAG](https://arxiv.org/abs/2505.20871) | Sun et al. | ACL | 2025 |

---

## §7 Confidence-Based Risk Management (24 papers)

Confidence serves calibration, selectivity, and coverage for reliable deployment.

### Actionable Signals

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Language models (mostly) know what they know](https://arxiv.org/abs/2207.05221) | Kadavath et al. | Preprint | 2022 |
| [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback](https://arxiv.org/abs/2305.14975) | Tian et al. | EMNLP | 2023 |
| [Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations](https://arxiv.org/abs/2503.14477) | Ji et al. | Preprint | 2025 |
| [Black-Box Hallucination Detection via Consistency Under the Uncertain Expression](https://arxiv.org/abs/2509.21999) | Joo et al. | Preprint | 2025 |

### Hallucination Detection

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models](https://arxiv.org/abs/2303.08896) | Manakul et al. | EMNLP | 2023 |
| [Detecting hallucinations in large language models using semantic entropy](https://arxiv.org/abs/2302.09664) | Farquhar et al. | Nature | 2024 |
| [Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs](https://arxiv.org/abs/2406.15927) | Han et al. | ICML Workshop | 2024 |
| [INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](https://arxiv.org/abs/2402.03744) | Chen et al. | ICLR | 2024 |
| [The Internal State of an LLM Knows When It's Lying](https://arxiv.org/abs/2304.13734) | Azaria & Mitchell | EMNLP Findings | 2023 |

### Conformal Guarantees

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Conformal prediction with large language models for multi-choice question answering](https://arxiv.org/abs/2305.18404) | Kumar et al. | Preprint | 2023 |
| [Conformal Language Modeling](https://arxiv.org/abs/2306.10193) | Quach et al. | ICLR | 2024 |
| [Language Models with Conformal Factuality Guarantees](https://arxiv.org/abs/2402.10978) | Mohri & Hashimoto | ICML | 2024 |
| [Large language model validity via enhanced conformal prediction methods](https://arxiv.org/abs/2406.09714) | Cherian et al. | NeurIPS | 2024 |
| [Conformal Language Model Reasoning with Coherent Factuality](https://arxiv.org/abs/2505.17126) | Rubin-Toles et al. | ICLR | 2025 |
| [API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access](https://arxiv.org/abs/2403.01216) | Su et al. | EMNLP Findings | 2024 |
| [Does confidence calibration improve conformal prediction?](https://arxiv.org/abs/2402.04344) | Xi et al. | TMLR | 2025 |

### Abstention & Alignment

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Self-Evaluation: Token Self-Eval](https://arxiv.org/abs/2312.09300) | Ren et al. | NeurIPS Workshop | 2023 |
| [Adaptation with self-evaluation to improve selective prediction in llms](https://arxiv.org/abs/2310.11689) | Chen et al. | EMNLP Findings | 2023 |
| [Selectively answering ambiguous questions](https://arxiv.org/abs/2305.14613) | Cole et al. | EMNLP | 2023 |
| [R-Tuning: Instructing Large Language Models to Say 'I Don't Know'](https://arxiv.org/abs/2311.09677) | Zhang et al. | NAACL | 2024 |
| [Fine-Tuning Large Language Models to Appropriately Abstain with Semantic Entropy](https://arxiv.org/abs/2410.17234) | Tjandra et al. | NeurIPS Workshop | 2024 |
| [ConfTuner: Training Large Language Models to Express Their Confidence Verbally](https://arxiv.org/abs/2508.18847) | Li et al. | NeurIPS | 2025 |
| [Teaching LLMs to Abstain via Fine-Grained Semantic Confidence Reward](https://arxiv.org/abs/2510.24020) | An & Xu | Preprint | 2025 |
| [Confidence-Based Response Abstinence: Improving LLM Trustworthiness via Activation-Based Uncertainty Estimation](https://arxiv.org/abs/2510.13750) | Huang et al. | UncertaiNLP | 2025 |

---

## §8 Confidence in Agentic Systems (14 papers)

Confidence propagates across tools, steps, and agents in composed action loops.

### Selective Escalation

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](https://arxiv.org/abs/2511.11306) | Fan et al. | Preprint | 2025 |
| [Know What You Don't Know: Uncertainty Calibration of Process Reward Models](https://arxiv.org/abs/2506.09338) | Park et al. | NeurIPS | 2025 |
| [Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning](https://arxiv.org/abs/2408.03314) | Snell et al. | ICLR (Oral) | 2025 |
| [Verification-Aware Planning for Multi-Agent Systems](https://arxiv.org/abs/2510.17109) | Xu et al. | EACL | 2026 |

### Self-Correction

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification](https://arxiv.org/abs/2502.14565) | Lee et al. | ICML | 2025 |
| [SSR: Socratic Self-Refine for Large Language Model Reasoning](https://arxiv.org/abs/2511.10621) | Shi et al. | Preprint | 2025 |
| [BacktrackAgent: Enhancing GUI Agent with Error Detection and Backtracking Mechanism](https://arxiv.org/abs/2505.20660) | Wu et al. | EMNLP | 2025 |

### Verifier-Guided Search

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models](https://arxiv.org/abs/2310.04406) | Zhou et al. | ICML | 2024 |
| [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816) | Zhang et al. | Preprint | 2024 |
| [Adaptive Uncertainty-Aware Tree Search for Robust Reasoning](https://arxiv.org/abs/2602.06493) | Song et al. | Preprint | 2026 |
| [AgentRM: Enhancing Agent Generalization with Reward Modeling](https://arxiv.org/abs/2502.18407) | Xia et al. | ACL | 2025 |
| [Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems](https://arxiv.org/abs/2502.19328) | Peng et al. | Preprint | 2025 |

### Multi-Agent Deliberation

| Paper | Authors | Venue | Year |
|-------|---------|-------|------|
| [ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs](https://arxiv.org/abs/2309.13007) | Chen et al. | ACL | 2024 |
| [Enhancing Multi-Agent Debate System Performance via Confidence Expression](https://arxiv.org/abs/2509.14034) | Lin & Hooi | EMNLP Findings | 2025 |

---

## 🔓 Open Challenges

1. **Heterogeneous Confidence Semantics** — Different signals measure different things; combining them without a shared scale is brittle.
2. **Composition Across Units and Horizons** — No general local→global propagation rules exist for composing step-level confidence into trajectory-level reliability.
3. **Source Attribution and Confidence Fusion** — Provenance is lost during fusion; knowing which source drove a decision matters for debugging and trust.
4. **Decision-Aware Objectives and Evaluation** — Metrics should match downstream actions, not just calibration or AUROC.
5. **Robustness and Portability** — Transfer of confidence mechanisms across models, prompts, and distributions remains fragile.

---

## 📝 Citation

```bibtex
@misc{li2026confidence,
  title  = {Confidence as Control: A Survey of Confidence Utilization
            in Large Language Models},
  author = {Yubo Li and Tianyang Zhou and Xiaobin Shen and
            Yidi Miao and Rema Padman and Ramayya Krishnan},
  year   = {2026},
  note   = {Preprint},
  url    = {https://yubol-bobo.github.io/assets/pdf/Conf_Survey.pdf}
}
```

## 🌐 Website

The companion project website is hosted at: **[https://yubol-bobo.github.io/awesome-llm-confidence/](https://yubol-bobo.github.io/awesome-llm-confidence/)**

## License

This project is for academic and educational purposes.
