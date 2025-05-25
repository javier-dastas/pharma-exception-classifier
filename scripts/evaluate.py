import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from sklearn.metrics import classification_report
import os

# Load data

print('Data loading ...')

# df = pd.read_csv('../data/exceptions-for-evaluation.csv')
df = pd.read_csv('../data/exceptions.csv')
label_list = sorted(df['label'].unique().tolist())
label_dict = {label: idx for idx, label in enumerate(label_list)}
inverse_label_dict = {v: k for k, v in label_dict.items()}
df['label_encoded'] = df['label'].map(label_dict)

print('Load the model ...')

# Load tokenizer and model from saved directory
model_path = '../models/bert-exception-classifier'
if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found. Please train the model first.")

tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

print('Tokenize all data ...')

# Tokenize all data
encodings = tokenizer(df['exception_text'].tolist(), truncation=True, padding=True, return_tensors='pt')

# Predict
model.eval()
with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=1)

# Map predictions
df['predicted_label'] = predictions.numpy()
df['predicted_label_name'] = df['predicted_label'].map(inverse_label_dict)


print('Save results ...')
# Save to CSV
# os.makedirs("outputs", exist_ok=True)
output_path = "../outputs/bert_predictions.csv"
df.to_csv(output_path, index=False)
print(f"Predictions saved to: {output_path}")

# Evaluation report
print("\nClassification Report:")
print(classification_report(df['label_encoded'], df['predicted_label'], target_names=label_list))
