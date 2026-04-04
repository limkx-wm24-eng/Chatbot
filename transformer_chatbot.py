import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
import numpy as np

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("faq_dataset.csv")

# Remove blank rows if you used a spaced viewing file by mistake
df = df.dropna()
df = df[(df["text"].astype(str).str.strip() != "") & (df["intent"].astype(str).str.strip() != "")]

# =========================
# 2. Encode labels
# =========================
labels = sorted(df["intent"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["label"] = df["intent"].map(label2id)

# Keep only needed columns
df = df[["text", "label"]]

# =========================
# 3. Convert to HF Dataset
# =========================
dataset = Dataset.from_pandas(df)

# Split train/test
split = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = split["train"]
test_ds = split["test"]

# =========================
# 4. Tokenizer + model
# =========================
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

# =========================
# 5. Metrics
# =========================
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels_true = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels_true)["accuracy"]
    precision = precision_metric.compute(predictions=predictions, references=labels_true, average="macro")["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels_true, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels_true, average="macro")["f1"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# =========================
# 6. Training config
# =========================
training_args = TrainingArguments(
    output_dir="./transformer_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    report_to="none",
)

# =========================
# 7. Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print("\nTransformer Evaluation Results:")
for k, v in results.items():
    print(f"{k}: {v}")

# =========================
# 8. Simple chatbot loop
# =========================
print("\nUniversity FAQ Chatbot (Transformer)")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye.")
        break

    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred_id = int(outputs.logits.argmax(dim=-1).item())
    intent = id2label[pred_id]

    responses = {
        "admission": "You can apply through the university admission portal and submit the required documents.",
        "fees": "You can check tuition fees and payment deadlines through the finance office or student portal.",
        "courses": "The university offers programmes such as IT, Business, and other diploma or degree courses.",
        "timetable": "You can view your class timetable through the student portal.",
        "contact": "You can contact the university through the official email or phone number listed on the website.",
        "greeting": "Hello. How can I help you today?",
        "thanks": "You are welcome.",
        "goodbye": "Goodbye. Have a nice day."
    }

    print("Predicted intent:", intent)
    print("Bot:", responses.get(intent, "Sorry, I do not understand your question."))