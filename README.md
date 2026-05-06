# GRPO Fine-tuning of Gemma 2B on SVAMP Math Word Problems

This repository contains code to fine-tune Google's **Gemma 2B** instruction-tuned model on the **SVAMP** (Symmetric Variability and Multi-solution Problem) dataset using **GRPO (Group Relative Policy Optimization)** – a reinforcement learning from human feedback (RLHF) variant from the TRL library.

The model learns to generate step-by-step reasoning inside `<think>` tags and a final numeric answer inside `<answer>` tags. After GRPO training, accuracy on the SVAMP test set improved from **43% to 73%**.

## 📊 Results

| Stage      | Accuracy (30 samples) |
|------------|-----------------------|
| Before GRPO | 43.3%                  |
| After GRPO  | 73.3%                  |

![Accuracy comparison](accuracy_comparison.png)

## 🧠 Method

- **Base model**: `google/gemma-2-2b-it` (4-bit quantized with NF4)
- **Dataset**: [ChilleD/SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP) (700 train, 300 test)
- **Fine‑tuning**: LoRA (rank=16, target modules: attention and MLP layers)
- **RL algorithm**: GRPO (Group Relative Policy Optimization) from 🤗 TRL
- **Reward functions**:
  - *Format reward*: encourages correct `<think>`/`<answer>` structure and no extra text.
  - *Correctness reward*: gives positive reward for exact or numerically close answers, negative for missing/wrong answers.

Training ran for 2 epochs (350 steps) with `num_generations=4`, `beta=0.005`, and a learning rate of 8e-6.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/gemma-grpo-svamp.git
cd gemma-grpo-svamp
