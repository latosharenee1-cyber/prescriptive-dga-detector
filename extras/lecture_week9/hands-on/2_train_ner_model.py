# Filename: 2_train_ner_model.py

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

# ----------------------------------------------------
# 1. Load dataset
# ----------------------------------------------------
raw_datasets = load_dataset('json', data_files='threat_reports.json')

# Example label list (adjust if your JSON has different label IDs)
label_list = ["O", "B-THREAT", "I-THREAT"]
label_to_id = {label: i for i, label in enumerate(label_list)}

# ----------------------------------------------------
# 2. Load tokenizer
# ----------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# ----------------------------------------------------
# 3. Tokenization & label alignment
# ----------------------------------------------------
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",  # <-- Added padding
        max_length=128,  # <-- Reasonable fixed length
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore padding
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])  # Same label for subwords
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)

# ----------------------------------------------------
# 4. Train / eval split
# ----------------------------------------------------
split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# ----------------------------------------------------
# 5. Load model
# ----------------------------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list),
    id2label={i: label for i, label in enumerate(label_list)},
    label2id=label_to_id
)

# ----------------------------------------------------
# 6. Training arguments
# ----------------------------------------------------
args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # evaluation every epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# ----------------------------------------------------
# 7. Trainer
# ----------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# ----------------------------------------------------
# 8. Train & save model
# ----------------------------------------------------
trainer.train()
print("✅ Model training complete.")
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")
print("✅ Model and tokenizer saved to ./results")
