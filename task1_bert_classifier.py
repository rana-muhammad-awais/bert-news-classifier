"""
Task 1: News Topic Classifier Using BERT

Objective: Fine-tune a transformer model (BERT) to classify news headlines into topic categories.
Dataset: AG News Dataset (Available on Hugging Face Datasets)

Instructions:
● Tokenize and preprocess the dataset
● Fine-tune the bert-base-uncased model using Hugging Face Transformers
● Evaluate the model using accuracy and F1-score
● Deploy the model using Streamlit or Gradio for live interaction
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
 BertTokenizer, 
 BertForSequenceClassification, 
 Trainer, 
 TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
 preds = np.argmax(p.predictions, axis=1)
 acc = accuracy_score(p.label_ids, preds)
 f1 = f1_score(p.label_ids, preds, average='weighted')
 return {'accuracy': acc, 'f1': f1}

def main():
 print("--- Task 1: News Topic Classifier Using BERT ---")
 
 # 1. Dataset Loading
 print("Loading AG News dataset from Hugging Face...")
 # Load a small subset for demonstration to save time and memory. 
 # Remove the 'split' argument to load the full dataset.
 dataset = load_dataset("ag_news")
 
 # Let's take a small subset for faster training during this internship demo
 # We take 2000 samples for training and 500 for testing
 train_dataset = dataset['train'].shuffle(seed=42).select(range(2000))
 test_dataset = dataset['test'].shuffle(seed=42).select(range(500))
 
 # 2. Tokenization and Preprocessing
 print("Loading BERT tokenizer...")
 tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
 
 def tokenize_function(examples):
 return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
 
 print("Tokenizing datasets...")
 tokenized_train = train_dataset.map(tokenize_function, batched=True)
 tokenized_test = test_dataset.map(tokenize_function, batched=True)
 
 # Format datasets for PyTorch
 tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
 tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
 
 # 3. Model Fine-Tuning
 print("Initializing BERT model for sequence classification...")
 model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
 
 training_args = TrainingArguments(
 output_dir='./results',
 num_train_epochs=3,
 per_device_train_batch_size=8,
 per_device_eval_batch_size=8,
 evaluation_strategy="epoch",
 save_strategy="epoch",
 learning_rate=2e-5,
 load_best_model_at_end=True,
 )
 
 trainer = Trainer(
 model=model,
 args=training_args,
 train_dataset=tokenized_train,
 eval_dataset=tokenized_test,
 compute_metrics=compute_metrics
 )
 
 print("Starting fine-tuning...")
 trainer.train()
 
 # 4. Evaluation
 print("Evaluating the model...")
 eval_results = trainer.evaluate()
 print(f"Evaluation Results: {eval_results}")
 
 # Save the final model and tokenizer
 save_directory = "./saved_bert_model"
 print(f"Saving model and tokenizer to {save_directory}...")
 model.save_pretrained(save_directory)
 tokenizer.save_pretrained(save_directory)
 print("Model saved successfully. It is ready for Streamlit deployment!")

if __name__ == "__main__":
 main()
