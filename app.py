from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = Flask(__name__)

# Load model and tokenizer
model_path = "roberta_ai_detector"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction endpoint


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return jsonify({
        "prediction": int(pred),
        "confidence": round(confidence, 4),
        "label": "AI" if pred == 1 else "Human"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
