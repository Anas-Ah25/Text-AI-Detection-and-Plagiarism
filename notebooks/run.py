from flask import Flask, request, jsonify, render_template
import torch
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)
import os
import time  # Import time for debugging purposes

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Load RoBERTa ========
start_time = time.time()
print("Loading RoBERTa model...")
roberta_model = RobertaForSequenceClassification.from_pretrained(
    "Roberta/Roberta/roberta_ai_detector/roberta_finetuned_model",
    local_files_only=True
).to(device)
roberta_tokenizer = RobertaTokenizer.from_pretrained(
    "Roberta/Roberta/roberta_ai_detector/roberta_finetuned_tokenizer",
    local_files_only=True
)
print(f"RoBERTa model loaded in {time.time() - start_time:.2f} seconds.")

# ======== Load ALBERT ========
start_time = time.time()
print("Loading ALBERT model...")
albert_model = AlbertForSequenceClassification.from_pretrained(
    "Albert2/albert_finetuned_model",
    local_files_only=True
).to(device)
albert_tokenizer = AlbertTokenizer.from_pretrained(
    "Albert2/albert_finetuned_tokenizer",
    local_files_only=True
)
print(f"ALBERT model loaded in {time.time() - start_time:.2f} seconds.")

# ======== Load DistilBERT ========
# Uncomment and modify if you plan to use DistilBERT later
start_time = time.time()
print("Loading DistilBERT model...")
distilbert_model =  DistilBertForSequenceClassification.from_pretrained(
    "DistilBERT\DistilBERT",
    local_files_only=True
).to(device)
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
    "DistilBERT/distil_finetuned_tokenizer",
    local_files_only=True
)
print(f"DistilBert model loaded in {time.time() - start_time:.2f} seconds.")

# ======== Voting Function ========
def predict_text(text):
    print("Starting prediction...")
    votes = []

    # RoBERTa
    print("Processing RoBERTa model...")
    start_time = time.time()
    inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = roberta_model(**{k: v.to(device) for k, v in inputs.items()})
    roberta_pred = torch.argmax(outputs.logits, dim=1).item()
    votes.append(roberta_pred)
    print(f"RoBERTa prediction took {time.time() - start_time:.2f} seconds.")

    # ALBERT
    print("Processing ALBERT model...")
    start_time = time.time()
    inputs = albert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = albert_model(**{k: v.to(device) for k, v in inputs.items()})
    albert_pred = torch.argmax(outputs.logits, dim=1).item()
    votes.append(albert_pred)
    print(f"ALBERT prediction took {time.time() - start_time:.2f} seconds.")

    # DistilBERT - Uncomment if using DistilBERT
    print("Processing DistilBERT model...")
    start_time = time.time()
    inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
         outputs = distilbert_model(**{k: v.to(device) for k, v in inputs.items()})
    distilbert_pred = torch.argmax(outputs.logits, dim=1).item()
    votes.append(distilbert_pred)
    print(f"DistilBERT prediction took {time.time() - start_time:.2f} seconds.")

    # Hard voting
    final_prediction = max(set(votes), key=votes.count)
    print(f"Final prediction (hard voting) completed.")
    return {
        "roberta": roberta_pred,
        "albert": albert_pred,
        "distilbert": distilbert_pred,  # Uncomment when using DistilBERT
        "final_prediction": final_prediction
    }

# ======== Flask Endpoints ========

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Favicon route to avoid 404
@app.route('/favicon.ico')
def favicon():
    return "", 204  # Empty response

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    print("Received prediction request...")
    start_time = time.time()

    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    result = predict_text(data["text"])
    print(f"Prediction completed in {time.time() - start_time:.2f} seconds.")
    return jsonify(result)

# ======== Run Flask App ========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
