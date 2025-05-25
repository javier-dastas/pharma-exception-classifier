# Pharma Exception Classifier

A BERT-based text classification project for pharmaceutical manufacturing exceptions.

## Features

- Fine-tunes BERT on exception reports.
- Predicts exception classes.
- Analyzes keyword correlations with classes.

## Setup

1. Install dependencies:

``` text
pip install -r requirements.txt
```

2. Prepare your dataset in `data/exceptions.csv`.

3. Train the model:

``` text
python scripts/train_model.py
```

4. Evaluate and analyze:

- Run `notebooks/bert_classification.ipynb` for evaluation.
- Run `scripts/analyze_keywords.py` for keyword analysis.
