#!/usr/bin/env python3
"""Upload script."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "/data/midas3/models/radtxt_small"
repo_url = "lrei/rad-small"

# Load the saved model and tokenizer using AutoModel and AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Push to Hugging Face Hub
model.push_to_hub(repo_url)
tokenizer.push_to_hub(repo_url)
