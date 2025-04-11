# Bioactivity Entity Extractor with QLoRA

This project is an end-to-end implementation of a lightweight LLM fine-tuned using QLoRA for extracting entities related to drug discovery from biomedical text.

### 🧪 Task:
Given a scientific abstract or patent sentence, extract:
- Drug name (compound)
- Target (e.g., protein or gene)
- Value (e.g., IC50)
- Assay type (e.g., cell line)

### 🛠 Tech Stack:
- PyTorch
- Hugging Face Transformers
- PEFT (QLoRA)
- Biomedical NER dataset (sample included)
- Evaluation: Precision, Recall, F1

### 📁 Structure:
- `data/` - Input dataset & tokenizer caches
- `models/` - Trained and saved models
- `scripts/` - Training, inference, evaluation
- `results/` - Output predictions & metrics
- `configs/` - Config files for model & training

To get started:
```bash
cd scripts
python train_qlora.py
```