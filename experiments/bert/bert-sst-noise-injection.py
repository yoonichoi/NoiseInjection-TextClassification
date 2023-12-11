from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import torch

# Load your data from a text file
file_name = "bert/train_orig.txt"
with open(file_name, "r", encoding="utf-8") as file:
    lines = file.readlines()
    labels, sentences = zip(*[line.strip().split("\t") for line in lines])
    labels = [int(label) for label in labels]

# Create a dictionary-based dataset
train_dataset = {"label": labels, "sentence": sentences}

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["sentence"], padding="max_length", truncation=True, return_tensors="pt")

tokenized_train_dataset = Dataset.from_dict(train_dataset).map(tokenize_function, batched=True)

with open("bert/test.txt", "r", encoding="utf-8") as file:
    lines_val = file.readlines()
    labels_val, sentences_val = zip(*[line.strip().split("\t") for line in lines_val])
    labels_val = [int(label) for label in labels_val]

# Create a dictionary-based dataset
val_dataset = {"label": labels_val, "sentence": sentences_val}

def tokenize_function_val(example):
    return tokenizer(example["sentence"], padding="max_length", truncation=True, return_tensors="pt")

tokenized_val_dataset = Dataset.from_dict(train_dataset).map(tokenize_function_val, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("sst2-finetuned-model",
                                  save_total_limit=2,)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=1)
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

evaluation_results = trainer.evaluate()
print(evaluation_results)

trainer.save_model("fine_tuned_sst2_model")
