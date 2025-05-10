# DistilBERT Classifier for AI/Human Text Detection

This folder contains:
- `distilbert_model.pth`: PyTorch model weights (classifier head only)
- `tokenizer/`: Tokenizer files needed to reproduce preprocessing
- Model: DistilBERT base uncased
- Training dataset: DAIGT (40k samples per class)
- Freeze strategy: Base model frozen, only classification head trained