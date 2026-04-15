# Awesome LLM Confidence Utilizations

[![Website](https://img.shields.io/badge/🌐_Project_Page-Visit-blue)](https://yubol-bobo.github.io/awesome-llm-confidence/)
[![Paper](https://img.shields.io/badge/📄_Paper-TMLR-green)]()
[![Awesome](https://img.shields.io/badge/Awesome-LLM_Confidence-orange)](https://github.com/yubol-bobo/awesome-llm-confidence)

> **Confidence as Control: A Survey of Confidence Utilization in Large Language Models**
>
> [Yubo Li](mailto:yubol@andrew.cmu.edu), Tianyang Zhou, Xiaobin Shen, Yidi Miao, Rema Padman, Ramayya Krishnan
>
> Carnegie Mellon University | TMLR 2025

## 📖 Overview

Most work on confidence in large language models has focused on **estimation**, uncertainty quantification, and calibration. In deployed systems, however, the key question is how confidence should be **used to govern behavior**.

This survey studies **confidence utilization**: the use of confidence-related signals to control system decisions. We formalize this through a unified framework in which confidence is defined over decision units under a local state and then consumed by a policy to determine actions.

<p align="center">
  <img src="assets/img/diagram.png" width="600" alt="Taxonomy">
</p>

## 🗂️ Taxonomy

We organize the literature across the **full LLM lifecycle**:

| Section | Domain | Key Actions |
|---------|--------|-------------|
| §3 | [**Confidence-Aware Training**](https://yubol-bobo.github.io/awesome-llm-confidence/pages/training.html) | Data curation, fine-tuning, distillation, preference optimization, RL |
| §4 | [**Confidence-Driven Inference**](https://yubol-bobo.github.io/awesome-llm-confidence/pages/inference.html) | Output selection, adaptive stopping, contrastive decoding |
| §5 | [**Confidence-Guided Model Selection**](https://yubol-bobo.github.io/awesome-llm-confidence/pages/routing.html) | Routing, cascading, deferral |
| §6 | [**Confidence-Gated RAG**](https://yubol-bobo.github.io/awesome-llm-confidence/pages/rag.html) | Retrieval gating, context filtering, groundedness, conformal RAG |
| §7 | [**Confidence-Based Risk Management**](https://yubol-bobo.github.io/awesome-llm-confidence/pages/risk.html) | Hallucination detection, conformal prediction, abstention |
| §8 | [**Confidence in Agentic Systems**](https://yubol-bobo.github.io/awesome-llm-confidence/pages/agentic.html) | Escalation, self-correction, verifier search, multi-agent debate |

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
- **Source**: self, sample-based, auxiliary, external, hybrid
- **Unit/Granularity**: token, claim, candidate, model, step, trajectory, episode
- **Functional Role**: selection, weighting, allocation, control-flow, aggregation, learning signal

## 🔓 Open Challenges

1. **Heterogeneous Confidence Semantics** — Different signals measure different things
2. **Composition Across Units and Horizons** — No general local→global propagation rules
3. **Source Attribution and Confidence Fusion** — Provenance lost during fusion
4. **Decision-Aware Objectives and Evaluation** — Metrics should match downstream actions
5. **Robustness and Portability** — Transfer across models, prompts, distributions

## 📊 Papers Covered

This survey covers **100+ methods** with detailed comparison tables across all six lifecycle stages. Visit the [project website](https://yubol-bobo.github.io/awesome-llm-confidence/) for a browsable, searchable collection of all papers organized by taxonomy.

## 📝 Citation

```bibtex
@article{li2025confidence,
  title={Confidence as Control: A Survey of Confidence Utilization 
         in Large Language Models},
  author={Li, Yubo and Zhou, Tianyang and Shen, Xiaobin 
          and Miao, Yidi and Padman, Rema and Krishnan, Ramayya},
  journal={Under review at Transactions on Machine Learning Research},
  year={2025}
}
```

## 🌐 Website

The companion project website is hosted at: **[https://yubol-bobo.github.io/awesome-llm-confidence/](https://yubol-bobo.github.io/awesome-llm-confidence/)**

## License

This project is for academic and educational purposes.
