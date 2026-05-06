# GRPO Fine-tuning of Gemma 2B on SVAMP Math Word Problems

This repository contains code to fine-tune Google's Gemma 2B instruction-tuned model on the SVAMP (Symmetric Variability and Multi-solution Problem) dataset using GRPO (Group Relative Policy Optimization), a reinforcement learning from human feedback (RLHF) variant from the TRL library.

The model learns to generate step-by-step reasoning inside `<think>` tags and a final numeric answer inside `<answer>` tags. After GRPO training, accuracy on the SVAMP test set improved from 43% to 73%.

## Results

| Stage        | Accuracy (30 samples) |
|--------------|-----------------------|
| Before GRPO  | 43.3%                 |
| After GRPO   | 73.3%                 |

Accuracy comparison bar chart and reward curves are generated during notebook execution.

## Method

- **Base model**: `google/gemma-2-2b-it` (4-bit quantized with NF4)
- **Dataset**: [ChilleD/SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP) (700 training, 300 test)
- **Fine-tuning**: LoRA (rank=16, target modules: attention and MLP layers)
- **RL algorithm**: GRPO (Group Relative Policy Optimization) from Hugging Face TRL
- **Reward functions**:
  - *Format reward*: encourages correct `<think>` and `<answer>` structure with no extraneous text.
  - *Correctness reward*: awards positive reward for exact or numerically close answers, negative for missing or incorrect answers.

Training ran for 2 epochs (350 steps) with `num_generations=4`, `beta=0.005`, and learning rate 8e-6.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/gemma-grpo-svamp.git
cd gemma-grpo-svamp
```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```
### 3. Run the notebook

Launch the provided Jupyter notebook:
```bash
jupyter notebook GRPO\ on\ Gemma\ 2B.ipynb
```
Execute all cells sequentially. The notebook will:
-Load the SVAMP dataset
-Format prompts with explicit tag requirements
-Load the 4-bit quantized Gemma 2B model
-Apply LoRA adapters
-Run GRPO training
-Evaluate model before and after training
-Plot reward curves and accuracy bar chart

### 4. Use the trained adapter (optional)

After training, the adapter is saved in ./grpo_gemma2_swamp_final. Load it with:

```bash
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto")
model = PeftModel.from_pretrained(model, "./grpo_gemma2_swamp_final")
```
