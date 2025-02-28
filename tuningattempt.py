from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import pandas as pd
import numpy as np
import evaluate

#Tokenizes the 'func' column for processing by the model with padding and truncation
def tokenize_function(data):
    return tokenizer(data["func"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#Loads the tokenizer and pretrained model for CodeBert
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2, torch_dtype="auto")

#Defines special parameters for the training loop to obeserve
#Currently only holds a location for the output directory when training completes
training_args = TrainingArguments(output_dir="test_trainer")

#Flags the loop to evaluate for 'accuracy'
metric = evaluate.load("accuracy")

#Loads a json file into dataset
dataset = load_dataset('json', data_files="function.json")
#Training requires a column named 'labels', 'target' in the original file functions as 'labels'
dataset = dataset.rename_column("target", "labels")
#Splits the dataset into a 90/10 split for training and evaluation
split_dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
#Tokenizes each of the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

#Create a Trainer object with the parameters defined above
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train_dataset,
    eval_dataset = tokenized_eval_dataset,
    compute_metrics = compute_metrics,
)

#Beings the training loop
trainer.train()
