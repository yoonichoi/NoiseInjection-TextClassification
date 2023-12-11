from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer

raw_datasets = load_dataset("glue", "sst2")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("sst2-finetuned-model",save_total_limit=2,)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)



def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=1)
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # Define your custom metrics function
    tokenizer=tokenizer,
)

trainer.train()

evaluation_results = trainer.evaluate(tokenized_datasets["validation"])
print(evaluation_results)

trainer.save_model("fine_tuned_sst2_model")