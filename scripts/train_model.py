import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load dataset
df = pd.read_csv('data/exceptions.csv')

# Encode labels
label_list = df['label'].unique().tolist()
label_dict = {label: idx for idx, label in enumerate(label_list)}
df['label_encoded'] = df['label'].map(label_dict)

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['exception_text'].tolist(),
    df['label_encoded'].tolist(),
    test_size=0.2,
    random_state=42
)

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create torch dataset
class ExceptionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = ExceptionDataset(train_encodings, train_labels)
val_dataset = ExceptionDataset(val_encodings, val_labels)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_list))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./models',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save model
model.save_pretrained('./models/bert-exception-classifier')
tokenizer.save_pretrained('./models/bert-exception-classifier')
