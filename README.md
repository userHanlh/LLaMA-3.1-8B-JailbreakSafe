
# LLaMA-3.1-8B-JailbreakSafe
Safety alignment for **LLaMA-3.1-8B** to defend against jailbreak attacks using **LoRA Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)**.

This project constructs multiple safety datasets and trains a jailbreak-resistant model while maintaining high usability for normal requests.

Model details:
[https://huggingface.co/hfuserh/LLaMA-3.1-8B-JailbreakSafe](https://huggingface.co/hfuserh/LLaMA-3.1-8B-JailbreakSafe)

---

# Overview

Large language models are vulnerable to **jailbreak prompts** that bypass safety alignment.

This project proposes a **safety alignment pipeline**:

1️⃣ **LoRA Supervised Fine-Tuning (SFT)**
2️⃣ **Direct Preference Optimization (DPO)**

The pipeline:

* builds jailbreak-related datasets
* trains a safety-aligned model
* evaluates jailbreak robustness and usability

---

# Repository Structure

```
LLaMA-3.1-8B-JailbreakSafe
│
├── llama_sft_dpo_jailbreaksafety
│   ├── data_process
│   │   ├── data_filter.py
│   │   └── generate_jailbreak_safe_answers.py
│   │
│   ├── eval.py
│   ├── lora_train.py
│   ├── dpo_train.py
│   └── val.py
│
├── lora_stage2_adapter_3epoch
│
├── lora_dpo_adapter
│
└── README.md
```

---

# Dataset Sources

This project uses jailbreak-related datasets from:

### WildJailbreak

AllenAI jailbreak dataset.

HuggingFace:
[https://huggingface.co/datasets/allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak)

---

### JailBench

Benchmark dataset for jailbreak evaluation.

GitHub:
[https://github.com/STAIR-BUPT/JailBench](https://github.com/STAIR-BUPT/JailBench)

---

# Data Processing

## 1. Harmless Dataset Filtering

File:

```
llama_sft_dpo_jailbreaksafety/data_process/data_filter.py
```

Purpose:

Constructs:

* **Boundary harmless dataset**
* **Pseudo-jailbreak harmless dataset**

Method:

* Uses **GPT-4o-mini** for filtering
* Source dataset: **WildJailbreak**

---

## 2. Harmful Dataset Construction

File:

```
llama_sft_dpo_jailbreaksafety/data_process/generate_jailbreak_safe_answers.py
```

Purpose:

Constructs:

* **Direct harmful dataset**
* **Jailbreak harmful dataset**

Method:

* Uses **GPT-5-mini** to generate safe responses
* Data sources:

  * WildJailbreak
  * JailBench

---

# Training

## LoRA SFT Training

Script:

```
llama_sft_dpo_jailbreaksafety/lora_train.py
```
Best checkpoint:

```
lora_stage2_adapter_3epoch
```

---

## DPO Preference Alignment

Script:

```
llama_sft_dpo_jailbreaksafety/dpo_train.py
```
Final aligned model:

```
lora_dpo_adapter
```

---

# Inference

Script:

```
llama_sft_dpo_jailbreaksafety/val.py
```

Used for:

* inference after SFT
* inference after DPO alignment

---

# Evaluation

Script:

```
llama_sft_dpo_jailbreaksafety/eval.py
```

Evaluation metrics:

* **Jailbreak Success Rate**
* **Answer Usability**

Evaluation model:

* **GPT-4o-mini**

---

# Results

| Model          | Jailbreak Success Rate | Normal Query Usability |
| -------------- | ---------------------- | ---------------------- |
| LoRA-SFT       | **1.45%**              | **88.57%**             |
| LoRA-SFT + DPO | **2.75%**              | **93.81%**             |

Key findings:

* LoRA-SFT significantly reduces jailbreak success rate
* DPO improves normal query usability while maintaining low jailbreak success

---


# Citation

If you use this project or datasets in your research, please cite the following works.

### WildTeaming / WildJailbreak Dataset

```bibtex
@misc{wildteaming2024,
  title={WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models},
  author={Liwei Jiang and Kavel Rao and Seungju Han and Allyson Ettinger and Faeze Brahman and Sachin Kumar},
  year={2024},
  eprint={2406.18510},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2406.18510}
}
````

### JailBench

```bibtex
@article{liu2025jailbench,
  title   = {JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models},
  author  = {Shuyi Liu and Simiao Cui and Haoran Bu and Yuming Shang and Xi Zhang},
  year    = {2025},
  journal = {arXiv preprint arXiv:2502.18935}
}
```

We thank the authors of these datasets for making their work publicly available.

---

# License

This project follows the licenses of the original datasets and models.

If you find this project useful, please consider giving it a star ⭐

```




